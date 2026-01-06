import discord
import os
import asyncio
import struct
import numpy as np
from groq import Groq
import edge_tts
from dotenv import load_dotenv
from discord.ext import commands
from discord.sinks import Sink, Filters
import sys
import traceback
import shutil
import ctypes.util
import warnings
from collections import deque
import json
import time
import wave
import uuid
import io
import logging
from openai import AsyncOpenAI 


print("System Version: 1.1.2 (PROD_DEBUG)")
warnings.filterwarnings("ignore", category=ResourceWarning)


logging.getLogger("discord.client").setLevel(logging.CRITICAL)
logging.getLogger("discord.gateway").setLevel(logging.CRITICAL)

class LogColors:
    INFO = '\033[94m'
    SUCCESS = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    RESET = '\033[0m'

def log(level, tag, message, color=LogColors.INFO):
    sys.stdout.write("\r" + " " * 80 + "\r")
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {color}[{level}] [{tag}] {message}{LogColors.RESET}")


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
KEYWORD_PATH = "Ka-Me-Ko_en_linux_v4_0_0.ppn"

log("INIT", "ENV", "Loading environment variables...")

if not GROQ_API_KEY: 
    log("FATAL", "CONFIG", "GROQ_API_KEY missing", LogColors.ERROR); exit()
if not DISCORD_TOKEN: 
    log("FATAL", "CONFIG", "DISCORD_TOKEN missing", LogColors.ERROR); exit()
if not OPENROUTER_API_KEY: 
    log("FATAL", "CONFIG", "OPENROUTER_API_KEY missing", LogColors.ERROR); exit()

if shutil.which("ffmpeg") is None:
    log("FATAL", "DEPENDENCY", "FFmpeg not found in PATH", LogColors.ERROR); exit()
else:
    log("READY", "DEPENDENCY", "FFmpeg verified", LogColors.SUCCESS)

if not discord.opus.is_loaded():
    try:
        opus_path = ctypes.util.find_library("opus")
        if opus_path:
            discord.opus.load_opus(opus_path)
            log("READY", "OPUS", "Audio library loaded", LogColors.SUCCESS)
    except Exception:
        log("WARN", "OPUS", "Audio library not found", LogColors.WARNING)


try:
    ai_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={"HTTP-Referer": "https://discord.com", "X-Title": "Kameko Bot"}
    )
    
    stt_client = AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY
    )

    bot = commands.Bot(command_prefix="!", intents=discord.Intents.all())
    log("READY", "CORE", "AI clients and Bot initialized", LogColors.SUCCESS)
except Exception as e:
    log("FATAL", "INIT", f"Initialization failed: {e}", LogColors.ERROR)
    exit()

chat_history = deque(maxlen=15)
event_stream = []

HALLUCINATIONS = [
    "продолжение следует", "субтитры", "subtitles", 
    "видео создано", "watching", "mbc", 
    "амедиатека", "перевод", "автор сценария"
]


async def get_ai_decision(history_items):
    log("AI", "DECISION", f"Analyzing {len(history_items)} events...")
    dialog_text = ""
    for item in history_items:
        dialog_text += f"- {item['user']}: {item['text']}\n"
    
    system_prompt = (
        "You are Kameko. Character: ironic, bold. "
        "RESPONSE FORMAT (JSON ONLY):\n"
        "{\n"
        '  "thought": "reasoning",\n'
        '  "should_speak": true/false,\n'
        '  "response": "text"\n'
        "}"
    )

    try:
        completion = await ai_client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CHAT LOG:\n{dialog_text}\n\nDecision?"}
            ],
            temperature=0.7,
            max_tokens=200,
        )

        result_content = completion.choices[0].message.content
        

        if "```" in result_content:
            result_content = result_content.replace("```json", "").replace("```", "").strip()
        
        data = json.loads(result_content)
        return data
        
    except Exception as e:

        log("ERROR", "AI_LLM", f"Request failed: {e}", LogColors.ERROR)
        return {"should_speak": False, "response": ""}

async def text_to_speech(text):
    log("TTS", "GENERATE", f"Synthesizing voice for phrase: {text[:30]}...")
    filename = "response.mp3"
    try:
        communicate = edge_tts.Communicate(text, "ru-RU-SvetlanaNeural", pitch="+45Hz", rate="+15%")
        await communicate.save(filename)
        return filename
    except Exception as e:
        log("ERROR", "TTS", f"Synthesis failed: {e}", LogColors.ERROR)
        return None

async def transcribe_audio(audio_data):
    if not audio_data: return ""
    try:
        
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        
        try:
            audio_np = audio_np.reshape(-1, 2)
            audio_mono = audio_np.mean(axis=1).astype(np.int16)
        except ValueError:
           
            audio_mono = audio_np

        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, 'wb') as wf:
            wf.setnchannels(1) 
            wf.setsampwidth(2) 
            wf.setframerate(48000)
            wf.writeframes(audio_mono.tobytes())
        
        audio_buffer.seek(0)
        
      
        transcription = await stt_client.audio.transcriptions.create(
            file=("voice.wav", audio_buffer.read()), 
            model="whisper-large-v3",
            language="ru",
            prompt="разговорный, сленг, без субтитров"
        )
        
        text = transcription.text.strip()
        log("STT", "GROQ", f"Transcription: '{text[:50]}...'", LogColors.SUCCESS)
        return text
    except Exception as e:
        log("ERROR", "STT", f"Transcription failed: {e}", LogColors.ERROR)
        return ""

# --- SINK ---
class UserStream:
    def __init__(self, user_id):
        self.user_id = user_id
        self.buffer = bytearray()
        self.is_recording = False
        self.silence_counter = 0
        self.recording_frames = 0

class MultiTrackSink(Sink):
    def __init__(self, voice_client, bot):
        super().__init__()
        self.vc = voice_client
        self.bot = bot
        self.user_streams = {}
        
        self.recording_frames = 0

        # --- НАСТРОЙКИ VAD ---
        self.volume_threshold = 500 # Порог громкости 
        self.silence_limit = 25    # Ждать 50 пакетов тишины (50 * 20мс = 1 секунда)
        self.min_recording_size = 100000 # Минимальный размер фразы (около 0.5 сек), чтобы не слать "пшики"
        
        log("INFO", "SINK", f"Sink active (Threshold: {self.volume_threshold}, Silence Wait: {self.silence_limit})")

    @Filters.container
    def write(self, pcm_audio, user):
        if not pcm_audio or len(pcm_audio) < 100: return
        
        # Если юзер не определился, используем ID 'unknown', чтобы не крашилось
        user_id = str(user) if user else "unknown_user"
        
        if user_id not in self.user_streams:
            self.user_streams[user_id] = UserStream(user_id)
            user_name = user.name if hasattr(user, 'name') else f"User_{user_id}"
            log("INFO", "SINK", f"New stream: {user_name}")

        stream = self.user_streams[user_id]
        
        try:
            # Анализ громкости
            audio_np = np.frombuffer(pcm_audio, dtype=np.int16)
            volume = np.abs(audio_np).mean()

            if volume > 100 and self.recording_frames % 20 == 0:
                 print(f"User: {user} | Volume: {volume:.2f} | Threshold: {self.volume_threshold}")
            self.recording_frames += 1

            # --- ЛОГИКА ЗАПИСИ ---
            
            # 1. Если слышим громкий звук
            if volume > self.volume_threshold:
                if not stream.is_recording:
                    log("VAD", "START", f"User {user_id} speaking (Vol: {volume:.0f})", LogColors.WARNING)
                    stream.is_recording = True
                    stream.buffer = bytearray()
                
                stream.silence_counter = 0 
            
            # 2. Если идет запись, но сейчас тихо 
            elif stream.is_recording:
                stream.silence_counter += 1

            # --- СОХРАНЕНИЕ И ПРОВЕРКИ ---
            if stream.is_recording:
                stream.buffer.extend(pcm_audio)
                
                # ПРОВЕРКА 1: Длительность тишины (Конец фразы)
                is_silence_timeout = stream.silence_counter > self.silence_limit
                
                # ПРОВЕРКА 2: Слишком длинная запись (Защита от вечного шума)
                # 48000 Гц * 2 байта * 2 канала * 10 секунд ≈ 1.9 МБ (грубо)
                # Если буфер больше 2 МБ, принудительно режем
                is_too_long = len(stream.buffer) > 2_000_000 
                
                if is_silence_timeout or is_too_long:
                    stream.is_recording = False # Останавливаем
                    
                    if is_too_long:
                        log("VAD", "CUT", "Phrase too long, forced cut.", LogColors.WARNING)

                    # Проверяем минимальную длину (чтобы не слать клики мышкой)
                    if len(stream.buffer) > self.min_recording_size:
                        log("VAD", "STOP", f"Phrase captured. Size: {len(stream.buffer)}b")
                        
                        data_copy = stream.buffer[:]
                        user_name = user.name if hasattr(user, 'name') else f"User_{user_id}"
                        
                        self.bot.loop.create_task(self.process_user_phrase(data_copy, user_name))
                    
                    # Очистка
                    stream.buffer = bytearray()
                    stream.silence_counter = 0
                    
        except Exception as e:
            log("ERROR", "SINK", f"Write error: {e}", LogColors.ERROR)

    async def process_user_phrase(self, audio_bytes, user_name):
        # Эта функция у тебя уже есть, но проверь её на всякий случай
        text = await transcribe_audio(audio_bytes)
        if not text or len(text) < 2: return
        
        text_lower = text.lower().strip()
        for bad_phrase in HALLUCINATIONS:
            if bad_phrase in text_lower:
                return

        event = {"id": uuid.uuid4().hex, "time": time.time(), "user": user_name, "text": text}
        event_stream.append(event)
        log("STREAM", "UPDATE", f"[{user_name}]: {text}", LogColors.SUCCESS)
        if len(event_stream) > 10: event_stream.pop(0)

    async def process_user_phrase(self, audio_bytes, user_name):
        text = await transcribe_audio(audio_bytes)
        if not text or len(text) < 2: return
        
        text_lower = text.lower().strip()
        for bad_phrase in HALLUCINATIONS:
            if bad_phrase in text_lower:
                log("INFO", "FILTER", "Hallucination detected, skipping.")
                return

        event = {"id": uuid.uuid4().hex, "time": time.time(), "user": user_name, "text": text}
        event_stream.append(event)
        log("STREAM", "UPDATE", f"Event added: [{user_name}]: {text}", LogColors.SUCCESS)
        if len(event_stream) > 10: event_stream.pop(0)

async def brain_loop(vc):
    global event_stream
    log("INFO", "BRAIN", "Engine running. Monitoring event stream.")
    last_processed_id = None

    while True:
        try:
            await asyncio.sleep(2)
            if not event_stream: continue
            latest_event = event_stream[-1]

            if latest_event['id'] == last_processed_id: continue

            log("BRAIN", "PROCESS", f"New event detected (ID: {latest_event['id']})", LogColors.SUCCESS)
            last_processed_id = latest_event['id']
            current_time = time.time()
            recent_events = [e for e in event_stream if current_time - e['time'] < 20]

            decision = await get_ai_decision(recent_events)
            if decision.get("should_speak"):
                response = decision.get("response")
                log("AI", "ACTION", f"Speaking: {response} | Reason: {decision.get('thought')}", LogColors.WARNING)
                
                audio_path = await text_to_speech(response)
                if audio_path and vc and vc.is_connected():
                    while vc.is_playing(): await asyncio.sleep(0.5)
                    vc.play(discord.FFmpegPCMAudio(audio_path))
                    while vc.is_playing(): await asyncio.sleep(1)
                    try: os.remove(audio_path)
                    except: pass
            else:
                log("AI", "IDLE", f"Decision: Silent | Reason: {decision.get('thought')}")
        except Exception as e:
            log("ERROR", "BRAIN", f"Runtime error: {e}", LogColors.ERROR)
            await asyncio.sleep(5)

@bot.event
async def on_ready():
    log("SYSTEM", "READY", f"Authenticated as {bot.user}", LogColors.SUCCESS)

@bot.command()
async def join(ctx):
    if ctx.author.voice:
        try:
            channel = ctx.author.voice.channel
            vc = await channel.connect(timeout=60.0, reconnect=True)
            vc.start_recording(MultiTrackSink(vc, bot), lambda *args: None)
            bot.loop.create_task(brain_loop(vc))
            await ctx.send("Connected. Monitoring active.")
        except Exception as e:
            log("ERROR", "JOIN", f"Connection failed: {e}", LogColors.ERROR)
            await ctx.send(f"Error: {e}")
    else:
        await ctx.send("You must be in a voice channel.")

@bot.command()
async def leave(ctx):
    if ctx.voice_client:
        log("INFO", "LEAVE", "Disconnecting...")
        ctx.voice_client.stop_recording()
        await ctx.voice_client.disconnect()

if __name__ == "__main__":
    try:
        bot.run(DISCORD_TOKEN)
    except Exception as e:
        log("FATAL", "CRASH", f"Bot crashed: {e}", LogColors.ERROR)

