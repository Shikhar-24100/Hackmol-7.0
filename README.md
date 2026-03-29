# 🇮🇳 Hindi Voice Assistant v5

An offline Hindi voice assistant built for Raspberry Pi. Understands spoken Hindi, classifies intent via a local LLM (Ollama), controls music playback, saves notes and reminders, detects mood, and can control an Android phone over ADB — all without internet.

---

## Table of Contents

- [Features](#features)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Configuration](#configuration)
- [Running the Assistant](#running-the-assistant)
- [Voice Commands Reference](#voice-commands-reference)
- [Mood Music Setup](#mood-music-setup)
- [ADB Phone Control Setup](#adb-phone-control-setup)
- [Data Storage](#data-storage)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [CLI Flags](#cli-flags)

---

## Features

| Category | What it does |
|---|---|
| **Speech Recognition** | Offline Hindi STT via Vosk |
| **Intent Classification** | Local LLM (llama3.2:1b) classifies every utterance into a structured intent |
| **Music Playback** | Play, stop songs from a local folder; fuzzy Hindi name matching |
| **Timers** | Multiple named concurrent timers; TTS fires on expiry |
| **Notes** | Save, recall by keyword, delete — persisted to disk |
| **Reminders** | Scheduled alerts with relative and absolute Hindi time expressions; persist across restarts |
| **Mood Music** | Detects emotional state from speech; picks songs from mood-tagged folders |
| **Phone Control** | ADB commands over USB or WiFi — open apps, scroll, screenshot, type text |
| **Chat** | General Hindi Q&A via llama3.2:3b; multi-turn memory |
| **TTS** | Offline speech synthesis via Piper |

---

## Hardware Requirements

- Raspberry Pi 4 (4 GB RAM minimum recommended)
- USB microphone or USB soundcard with microphone
- Speaker (3.5mm or HDMI)
- Android phone (optional, for ADB features)
- USB cable or same WiFi network as the phone (for ADB)

---

## Software Requirements

| Package | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Runtime |
| [Ollama](https://ollama.com) | latest | Local LLM server |
| [Vosk](https://alphacephei.com/vosk/) | 0.3.x | Offline Hindi STT |
| [Piper](https://github.com/rhasspy/piper) | latest | Offline TTS |
| pygame | 2.x | Audio/music playback |
| sounddevice | 0.4.x | Microphone capture |
| numpy | any | Audio buffer handling |
| requests | any | Ollama HTTP API |
| adb (Android SDK) | any | Phone control (optional) |

---

## Installation

### 1. Clone / copy the script

```bash
mkdir -p /home/kapil/hindi-assistant
cp hindi_assistant_v5.py /home/kapil/hindi-assistant/
cd /home/kapil/hindi-assistant
```

### 2. Install Python dependencies

```bash
pip install vosk sounddevice pygame numpy requests piper-tts
```

### 3. Install and start Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:1b
ollama pull llama3.2:3b
ollama serve &
```

### 4. Download the Vosk Hindi model

```bash
wget https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip
unzip vosk-model-small-hi-0.22.zip -d /home/kapil/hindi-assistant/vosk-hindi-small
```

### 5. Download a Piper Hindi voice

```bash
mkdir -p /home/kapil/hindi-assistant/voices/hi_IN-female
cd /home/kapil/hindi-assistant/voices/hi_IN-female
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/female/medium/hi_IN-female-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/female/medium/hi_IN-female-medium.onnx.json
```

### 6. Add songs

```bash
mkdir -p /home/kapil/hindi-assistant/songs_dir
cp *.mp3 /home/kapil/hindi-assistant/songs_dir/
```

### 7. (Optional) Install ADB

```bash
sudo apt install adb
```

---

## Directory Structure

```
/home/kapil/hindi-assistant/
├── hindi_assistant_v5.py
├── vosk-hindi-small/           # Vosk model folder
├── voices/
│   └── hi_IN-female/
│       ├── hi_IN-female-medium.onnx
│       └── hi_IN-female-medium.onnx.json
└── songs_dir/
    ├── bairan.mp3
    ├── tum hi ho.mp3
    ├── song_moods.json          # optional mood tags
    ├── sad/                     # optional mood subfolders
    │   └── *.mp3
    ├── happy/
    ├── romantic/
    ├── energetic/
    └── relaxed/

~/.hindi_assistant/              # auto-created at runtime
├── notes.json
└── reminders.json
```

---

## Configuration

All constants are at the top of the script:

| Constant | Default | Description |
|---|---|---|
| `VOSK_MODEL_PATH` | `/home/kapil/hindi-assistant/vosk-hindi-small` | Path to the Vosk Hindi model |
| `VOICES_DIR` | `./voices` (relative to script) | Directory containing Piper `.onnx` voice |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `SONGS_DIR` | `/home/kapil/hindi-assistant/songs_dir` | Root folder for MP3s |
| `DATA_DIR` | `~/.hindi_assistant` | Notes and reminders storage |
| `INTENT_MODEL` | `llama3.2:1b` | Fast model for intent classification |
| `CHAT_MODEL` | `llama3.2:3b` | Larger model for open-ended chat |
| `SAMPLE_RATE` | `16000` | Microphone sample rate (Hz) |
| `BLOCK_SIZE` | `4000` | Mic audio block size |

---

## Running the Assistant

```bash
# Standard run
python hindi_assistant_v5.py

# Disable text-to-speech (text output only)
python hindi_assistant_v5.py --no-speech

# Keyword-only mode (no Ollama required)
python hindi_assistant_v5.py --no-llm

# Custom songs directory
python hindi_assistant_v5.py --songs-dir /media/usb/songs

# Disable ADB phone control
python hindi_assistant_v5.py --no-adb

# Connect to phone over WiFi ADB
python hindi_assistant_v5.py --adb-wifi 192.168.1.100

# Combine flags
python hindi_assistant_v5.py --no-adb --songs-dir ~/music
```

---

## Voice Commands Reference

### Music

| Say | Action |
|---|---|
| `गाना बजाओ तुम ही हो` | Play song by name |
| `बैरन चलाओ` | Play song by Hindi alias |
| `बंद करो` / `रुको` / `stop` | Stop playback |

### Timers

| Say | Action |
|---|---|
| `5 मिनट का टाइमर लगाओ` | Set a 5-minute timer |
| `चाय का टाइमर 3 मिनट` | Named timer (fires "चाय का समय हो गया!") |
| `डेढ़ घंटे का टाइमर` | 90-minute timer |
| `1 घंटा 30 मिनट का टाइमर` | Compound duration |
| `टाइमर बंद करो` | Cancel all timers |
| `कितने टाइमर हैं` | List active timers with remaining time |

### Notes

| Say | Action |
|---|---|
| `नोट करो कल डॉक्टर के पास जाना है` | Save a note |
| `लिख लो पानी भरना है` | Save a note (alternate trigger) |
| `मेरे नोट्स सुनाओ` | Read all notes |
| `आखिरी नोट सुनाओ` | Read most recent note |
| `डॉक्टर वाला नोट सुनाओ` | Search notes by keyword |
| `सारे नोट्स मिटाओ` | Delete all notes |

### Reminders

| Say | Action |
|---|---|
| `2 घंटे बाद याद दिलाना पानी पीना है` | Relative reminder |
| `कल सुबह 8 बजे याद दिलाओ दवाई लेनी है` | Absolute reminder (tomorrow) |
| `शाम 6 बजे याद दिलाओ` | Absolute reminder (today evening) |
| `reminders बताओ` | List pending reminders |
| `reminder हटाओ` | Cancel all reminders |

**Supported time expressions:**

| Hindi | Meaning |
|---|---|
| `X मिनट बाद` | X minutes from now |
| `X घंटे बाद` | X hours from now |
| `डेढ़ घंटे बाद` | 90 minutes from now |
| `X दिन बाद` | X days from now |
| `[कल/परसों] [सुबह/शाम/रात] X बजे` | Absolute time on a specific day/period |

### Mood Music

| Say | Mood |
|---|---|
| `उदास हूँ कुछ लगाओ` | sad |
| `मस्त गाना चलाओ` | happy |
| `romantic mood है` | romantic |
| `डांस वाला गाना चलाओ` | energetic |
| `सुकून वाला गाना` | relaxed |

### Phone Control (ADB)

| Say | Action |
|---|---|
| `व्हाट्सएप खोलो` | Open WhatsApp |
| `यूट्यूब खोलो` | Open YouTube |
| `स्पॉटिफाई खोलो` | Open Spotify |
| `क्रोम खोलो` | Open Chrome |
| `कैमरा खोलो` | Open Camera |
| `सेटिंग खोलो` | Open Settings |
| `स्क्रीनशॉट लो` | Take screenshot → saved to `./screen.png` |
| `होम पर जाओ` | Press Home |
| `वापस जाओ` | Press Back |
| `स्क्रीन लॉक करो` | Lock screen |
| `स्क्रीन अनलॉक करो` | Unlock screen |
| `ऊपर स्क्रोल करो` | Scroll up |
| `नीचे स्क्रोल करो` | Scroll down |
| `फ़ोन की आवाज़ बढ़ाओ` | Volume up |
| `फ़ोन की आवाज़ घटाओ` | Volume down |
| `hello टाइप करो` | Type text on screen |

### General

| Say | Action |
|---|---|
| `भारत की राजधानी क्या है` | General Hindi Q&A |
| `quit` / `exit` / `बाहर निकलो` | Exit the assistant |

---

## Mood Music Setup

**Option A — Mood subfolders** (recommended):

```
songs_dir/
├── sad/
│   ├── tere bina.mp3
│   └── channa mereya.mp3
├── happy/
│   └── badtameez dil.mp3
├── romantic/
├── energetic/
└── relaxed/
```

**Option B — Tag existing songs** via `songs_dir/song_moods.json`:

```json
{
  "bairan":     ["sad", "romantic"],
  "tum hi ho":  ["romantic"],
  "badtameez":  ["happy", "energetic"]
}
```

Keys must match the MP3 filename stem (no extension, lowercase). If both a subfolder and `song_moods.json` exist, the subfolder takes priority. If neither matches, a random song from the full library is played as fallback.

---

## ADB Phone Control Setup

### USB

1. Enable **Developer Options** on the phone — tap Build Number 7 times in Settings → About Phone
2. Enable **USB Debugging** inside Developer Options
3. Connect via USB cable and accept the RSA key fingerprint prompt on the phone
4. Verify with `adb devices` — your device should appear with status `device`

### WiFi

```bash
# With USB connected first:
adb tcpip 5555
adb connect <phone-ip>:5555

# Or let the assistant connect automatically at startup:
python hindi_assistant_v5.py --adb-wifi 192.168.1.100
```

Find the phone IP at: Settings → About Phone → Status → IP Address

---

## Data Storage

All persistent data lives in `~/.hindi_assistant/` and is created automatically on first run.

**`notes.json`**
```json
[
  {
    "id": 1712345678000,
    "text": "कल डॉक्टर के पास जाना है",
    "created": "15/04/2025 10:30"
  }
]
```

**`reminders.json`**
```json
[
  {
    "id": 1712345679000,
    "message": "दवाई लेनी है",
    "trigger_iso": "2025-04-16T08:00:00",
    "done": false,
    "created": "2025-04-15T10:30:00"
  }
]
```

Both files are plain JSON and can be edited manually. Set `"done": true` on a reminder to disable it without deleting it. The background reminder thread checks every 20 seconds.

---

## Architecture

```
Microphone
    │
    ▼
Vosk STT ──────────────► normalize()
                              │
                 ┌────────────┴────────────┐
                 │                         │
          Keyword fast-path          LLM classifier
          (instant, no wait)         llama3.2:1b
                 │                         │
                 └────────────┬────────────┘
                              │
                        Intent + params
                              │
            ┌─────────────────┼──────────────────────┐
            │                 │                      │
       play_song        set_timer /           phone_command
       stop_song        save_note /            (ADB exec)
       mood_music       recall_notes /
                        set_reminder / etc.
                              │
                         chat intent
                              │
                     llm_reply present?
                      YES → speak (1b answered)
                      NO  → llama3.2:3b
                              │
                             TTS
                           (Piper)
                              │
                           Speaker
```

**Key design decisions:**

- The keyword fast-path runs before the LLM returns. For unambiguous commands (stop, play, phone keyword), the result is used immediately and latency is near-zero.
- `keep_alive=-1` on every Ollama call keeps both models pinned in RAM permanently — no reload penalty between turns.
- Timer expiry callbacks and reminder checks push messages to `_announce_q`. The main loop drains this queue before every `listen_once()` so TTS and microphone capture never overlap.
- Notes and reminders use `threading.Lock` and are flushed to JSON after every write operation.
- The 1b model answers simple chat questions inline via the `reply` JSON field. The 3b model only fires when `reply` is null, keeping average response latency low.
- Both LLM models are warmed up in parallel threads at startup to minimise the initial ready time.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No .onnx voice found` | Confirm `VOICES_DIR` contains a subfolder with a `.onnx` file (not `.json` only) |
| Vosk outputs wrong/garbled words | Switch to the full model `vosk-model-hi-0.22` for better accuracy at the cost of ~300 MB RAM |
| LLM timeout on every turn | Run `ollama ps` to confirm models are loaded. Increase `timeout=` in `classify_intent()` if the Pi is slow |
| `[ADB] No device` | Run `adb devices`. Accept the RSA fingerprint on the phone screen. For WiFi ADB confirm both are on the same subnet |
| Timer fires but no audio | `_announce_q` is only drained between voice turns, not during mic capture — this is by design |
| Reminder never fires | Check `trigger_iso` in `~/.hindi_assistant/reminders.json` is a valid ISO datetime and `done` is `false` |
| Song not found by Hindi name | Add an entry to `HINDI_SONG_ALIASES` mapping the Hindi word to the MP3 filename stem (without extension) |
| `pygame.error: No such file` | Confirm the full MP3 path printed at startup is accessible. Check file permissions |
| High CPU / thermal throttle on Pi | Set both `INTENT_MODEL` and `CHAT_MODEL` to `llama3.2:1b` and confirm `num_thread=4` |

---

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--no-speech` | off | Print AI replies to terminal only; skip Piper TTS |
| `--no-llm` | off | Keyword matching only; Ollama not started or required |
| `--no-adb` | off | Disable all ADB phone control features |
| `--songs-dir PATH` | value of `SONGS_DIR` constant | Override the songs directory at runtime |
| `--adb-wifi IP` | none | Connect to Android device over WiFi ADB on port 5555 at startup |
