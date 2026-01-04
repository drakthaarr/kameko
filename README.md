# kameko
Since this is for your CV/Portfolio, the README should emphasize **software architecture**, **asynchronous programming**, **API integration**, and **signal processing**. It should position the bot as an "AI Agent" rather than just a "Discord Bot."

---

# Multi-Agent Voice Interaction System (Kameko)

A high-performance, asynchronous Python application that implements a **Real-Time Speech-to-Speech (S2S) Pipeline**. This system integrates Voice Activity Detection (VAD), Neural Speech-to-Text, LLM-based decision logic, and Neural Text-to-Speech to facilitate natural, low-latency interactions in a multi-user environment.

## Architecture Overview

The system is built on a non-blocking event loop using `asyncio`, coordinating several specialized micro-services:

* **Audio Ingestion Engine:** A custom implementation of `discord.Sinks` that performs real-time PCM audio analysis using `NumPy` to identify voice activity via mean amplitude thresholds.
* **Speech-to-Text (STT) Pipeline:** Utilizes **Whisper-large-v3** via **Groq’s** inference engine for ultra-low latency transcription of Russian and English dialects.
* **Cognitive Logic (The "Brain"):** Powered by **Llama 3.3 70B** (via OpenRouter). It uses a "context-aware decision" model where the AI doesn't just respond to prompts but monitors a live event stream to determine *if* and *when* it is socially appropriate to intervene.
* **Neural Synthesis (TTS):** Leverages **Edge-TTS** (SvetlanaNeural) with customized pitch and rate parameters to produce human-like prosody.

---

## 🛠 Technical Stack

| Category | Technology |
| --- | --- |
| **Language** | Python 3.10+ |
| **Concurrency** | `asyncio`, Multi-threading for I/O |
| **Digital Signal Processing** | `NumPy`, `Wave`, `FFmpeg` |
| **APIs & LLMs** | OpenAI SDK, Groq, OpenRouter (Llama 3.3) |
| **Real-time Comms** | `discord.py`, WebSockets |

---

## Key Engineering Challenges Solved

### 1. Low-Latency Voice Activity Detection (VAD)

Developed a custom `UserStream` manager that monitors raw byte streams. By implementing a volume threshold and silence counter system, the application identifies the end of a "speech turn" without requiring a Push-To-Talk mechanism.

### 2. Hallucination Mitigation

Whisper models frequently hallucinate metadata (e.g., "Subtitles by...") during silent audio segments. I implemented a regex-based filtering layer and a minimum buffer threshold to ensure the AI only processes high-confidence human speech.

### 3. Contextual Decision Logic

Unlike traditional request-response bots, this system uses a `deque` (double-ended queue) to maintain a sliding window of the last 15 conversation events. The LLM receives this context and returns a structured JSON schema:

```json
{
  "thought": "Analysis of the current conversation flow",
  "should_speak": true,
  "response": "Synthesized output"
}

```

---

## Installation & Setup

### Prerequisites

* **FFmpeg** (Compiled with `libopus`)
* **Python 3.10+**

### Environment Configuration

The system requires a `.env` file with the following keys:

* `DISCORD_TOKEN`: Bot authentication.
* `GROQ_API_KEY`: For Whisper-v3-large inference.
* `OPENROUTER_API_KEY`: For Llama-3.3-70b reasoning.

---

## 📈 Potential Use Cases

* **Automated Moderation:** Real-time sentiment analysis of voice channels.
* **Interactive NPCs:** Dynamic characters for gaming communities.
* **Accessibility:** Real-time transcription and voice-assisted navigation for visually impaired users in digital workspaces.

---

**Would you like me to expand on the "Engineering Challenges" section or add a "Performance Metrics" section to further highlight your technical skills?**
