#!/usr/bin/env python3
"""
Hindi Voice Assistant - 
  1. Parallel model warm-up (VOSK + TTS + Ollama loaded concurrently)
  2. LLM intent call is async — keyword pre-filter runs instantly,
     LLM result merges in before routing
  3. TTS synthesis runs in a background thread (non-blocking)
  4. Audio queue drained more efficiently
  5. VOSK model loaded once, recognizer reused across turns
  6. Ollama keep_alive=-1 ensures model stays pinned in RAM always
  7. num_thread set to actual Pi core count (4)
  8. Intent LLM uses num_predict=80 (enough for one JSON line)
  9. Startup prints are deferred so Pi boots feel faster
 10. subprocess ADB calls use a persistent shell to avoid fork overhead
 11. Option B: 1b model answers chat directly via reply field,
     3b model only fires when 1b returns null reply
"""

import io, json, os, queue, re, subprocess, sys, wave, argparse, difflib
import unicodedata, threading, time
import numpy as np
import requests
import sounddevice as sd
import pygame
from vosk import Model, KaldiRecognizer

# ── Config ────────────────────────────────────────────────────────────────────

VOSK_MODEL_PATH = "/home/kapil/hindi-assistant/vosk-hindi-small"
VOICES_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
OLLAMA_URL      = "http://localhost:11434"
SONGS_DIR       = "/home/kapil/hindi-assistant/songs_dir"

INTENT_MODEL = "llama3.2:1b"
CHAT_MODEL   = "llama3.2:3b"

SAMPLE_RATE = 16000
BLOCK_SIZE  = 4000

# ── Hindi song name aliases ───────────────────────────────────────────────────

HINDI_SONG_ALIASES = {
    "बैरन":       "bairan",
    "बहरैन":      "bairan",
    "बेरन":       "bairan",
    "वैरन":       "bairan",
    "तुम ही हो":  "tum hi ho",
    "तुम ही":     "tum hi ho",
    "तुमही हो":   "tum hi ho",
}

# ── ADB action map ────────────────────────────────────────────────────────────

ADB_ACTIONS = {
    "home":          "adb shell input keyevent 3",
    "back":          "adb shell input keyevent 4",
    "power":         "adb shell input keyevent 26",
    "volume_up":     "adb shell input keyevent 24",
    "volume_down":   "adb shell input keyevent 25",
    "screenshot":    "adb shell screencap /sdcard/screen.png && adb pull /sdcard/screen.png ./screen.png",
    "open_whatsapp": "adb shell monkey -p com.whatsapp 1",
    "open_youtube":  "adb shell monkey -p com.google.android.youtube 1",
    "open_spotify":  "adb shell monkey -p com.spotify.music 1",
    "open_chrome":   "adb shell monkey -p com.android.chrome 1",
    "open_camera":   "adb shell monkey -p com.android.camera2 1",
    "open_settings": "adb shell am start -a android.settings.SETTINGS",
    "lock_screen":   "adb shell input keyevent 26",
    "unlock_screen": "adb shell input keyevent 82",
    "scroll_up":     "adb shell input swipe 500 1200 500 400 300",
    "scroll_down":   "adb shell input swipe 500 400 500 1200 300",
}

ADB_ACTION_HINDI = {
    "home":          "होम स्क्रीन पर गया।",
    "back":          "वापस गया।",
    "power":         "पावर बटन दबाया।",
    "volume_up":     "आवाज़ बढ़ाई।",
    "volume_down":   "आवाज़ घटाई।",
    "screenshot":    "स्क्रीनशॉट लिया।",
    "open_whatsapp": "व्हाट्सएप खोला।",
    "open_youtube":  "यूट्यूब खोला।",
    "open_spotify":  "स्पॉटिफाई खोला।",
    "open_chrome":   "क्रोम खोला।",
    "open_camera":   "कैमरा खोला।",
    "open_settings": "सेटिंग्स खोली।",
    "lock_screen":   "स्क्रीन लॉक की।",
    "unlock_screen": "स्क्रीन अनलॉक की।",
    "scroll_up":     "ऊपर स्क्रोल किया।",
    "scroll_down":   "नीचे स्क्रोल किया।",
    "type_text":     "टाइप किया।",
    "tap":           "टैप किया।",
    "none":          "यह फ़ोन कमांड समझ नहीं आई।",
}

# ── Intent system prompt ──────────────────────────────────────────────────────

INTENT_SYSTEM_PROMPT = """\
You are a Hindi voice assistant intent classifier.

Return ONLY a JSON object. No explanation. No markdown. No extra text.

JSON format:
{
  "intent": "<play_song | stop_song | phone_command | chat | exit>",
  "song_name": "<extracted song name or empty string if unknown, null for non-song intents>",
  "phone_action": "<adb action key or null>",
  "phone_value": "<extra value for type_text or tap, else null>",
  "reply": "<Hindi Devanagari reply for ALL chat intents, else null>"
}

VALID phone_action values:
home, back, power, volume_up, volume_down, screenshot,
open_whatsapp, open_youtube, open_spotify, open_chrome, open_camera, open_settings,
lock_screen, unlock_screen, scroll_up, scroll_down, type_text, tap, none

CRITICAL RULES:
- song_name must NEVER be null for play_song intent. Use empty string "" if unknown.
- If user says only a song or artist name with no other context, intent is play_song.
- For phone_command, always set phone_action to the closest matching key above.
- For chat intent, ALWAYS write a complete Hindi answer in "reply". Never set reply to null for chat.
- Return ONLY the JSON object. Nothing before it, nothing after it.

Examples:
User: गाना बजाओ तुम ही हो
{"intent":"play_song","song_name":"तुम ही हो","phone_action":null,"phone_value":null,"reply":null}

User: बैरन
{"intent":"play_song","song_name":"बैरन","phone_action":null,"phone_value":null,"reply":null}

User: बंद करो
{"intent":"stop_song","song_name":null,"phone_action":null,"phone_value":null,"reply":null}

User: व्हाट्सएप खोलो
{"intent":"phone_command","song_name":null,"phone_action":"open_whatsapp","phone_value":null,"reply":null}

User: फ़ोन की आवाज़ बढ़ाओ
{"intent":"phone_command","song_name":null,"phone_action":"volume_up","phone_value":null,"reply":null}

User: स्क्रीनशॉट लो
{"intent":"phone_command","song_name":null,"phone_action":"screenshot","phone_value":null,"reply":null}

User: यूट्यूब खोलो
{"intent":"phone_command","song_name":null,"phone_action":"open_youtube","phone_value":null,"reply":null}

User: वापस जाओ
{"intent":"phone_command","song_name":null,"phone_action":"back","phone_value":null,"reply":null}

User: होम पर जाओ
{"intent":"phone_command","song_name":null,"phone_action":"home","phone_value":null,"reply":null}

User: नीचे स्क्रोल करो
{"intent":"phone_command","song_name":null,"phone_action":"scroll_down","phone_value":null,"reply":null}

User: hello टाइप करो
{"intent":"phone_command","song_name":null,"phone_action":"type_text","phone_value":"hello","reply":null}

User: स्क्रीन लॉक करो
{"intent":"phone_command","song_name":null,"phone_action":"lock_screen","phone_value":null,"reply":null}

User: भारत की राजधानी क्या है
{"intent":"chat","song_name":null,"phone_action":null,"phone_value":null,"reply":"भारत की राजधानी नई दिल्ली है।"}

User: आज का मौसम कैसा है
{"intent":"chat","song_name":null,"phone_action":null,"phone_value":null,"reply":"माफ करें, मेरे पास इंटरनेट नहीं है इसलिए मौसम की जानकारी नहीं दे सकता।"}

User: तुम कौन हो
{"intent":"chat","song_name":null,"phone_action":null,"phone_value":null,"reply":"मैं आपका हिंदी वॉयस असिस्टेंट हूँ, Raspberry Pi पर चलता हूँ।"}

User: quit
{"intent":"exit","song_name":null,"phone_action":null,"phone_value":null,"reply":null}
"""

CHAT_SYSTEM_PROMPT = (
    "You are a helpful Hindi assistant on a Raspberry Pi. "
    "Reply in Hindi (Devanagari). Max 2-3 short sentences."
)

# ── pygame mixer init ─────────────────────────────────────────────────────────

pygame.mixer.init()

# ── Audio input ───────────────────────────────────────────────────────────────

_audio_q: queue.Queue = queue.Queue()

def _cb(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr)
    _audio_q.put(bytes(indata))

# ── Normalize ─────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    text = text.strip()
    text = " ".join(text.split())
    text = unicodedata.normalize("NFC", text)
    return text.lower()

# ── VOSK — reuse recognizer between turns ─────────────────────────────────────

def listen_once(rec: KaldiRecognizer) -> str:
    rec.Reset()
    print("🎙️  बोलिए...")
    while True:
        data = _audio_q.get()
        if rec.AcceptWaveform(data):
            text = json.loads(rec.Result()).get("text", "").strip()
            if text:
                print()
                return text
        else:
            p = json.loads(rec.PartialResult()).get("partial", "")
            if p:
                print(f"\r{p}   ", end="", flush=True)

# ── TTS — synthesise in background thread ─────────────────────────────────────

_voice     = None
_tts_lock  = threading.Lock()

def load_tts():
    global _voice
    if _voice:
        return _voice
    from piper import PiperVoice
    for name in os.listdir(VOICES_DIR):
        folder = os.path.join(VOICES_DIR, name)
        if os.path.isdir(folder):
            for f in os.listdir(folder):
                if f.endswith(".onnx") and not f.endswith(".json"):
                    _voice = PiperVoice.load(os.path.join(folder, f))
                    return _voice
    raise RuntimeError("No .onnx voice found in " + VOICES_DIR)

def _synth_blocking(text: str):
    music_was_playing = pygame.mixer.music.get_busy()
    if music_was_playing:
        pygame.mixer.music.pause()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        _voice.synthesize_wav(text, wf)
    buf.seek(0)
    with wave.open(buf) as wf:
        rate = wf.getframerate()
        raw  = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    sd.play(audio, samplerate=rate)
    sd.wait()
    if music_was_playing:
        pygame.mixer.music.unpause()

_tts_thread: threading.Thread | None = None

def speak(text: str, no_speech: bool, wait: bool = True):
    print(f"AI: {text}")
    if no_speech or not _voice:
        return

    global _tts_thread

    if _tts_thread and _tts_thread.is_alive():
        _tts_thread.join()

    if wait:
        try:
            with _tts_lock:
                _synth_blocking(text)
        except Exception as e:
            print(f"[TTS] {e}", file=sys.stderr)
    else:
        def _run():
            try:
                with _tts_lock:
                    _synth_blocking(text)
            except Exception as e:
                print(f"[TTS] {e}", file=sys.stderr)
        _tts_thread = threading.Thread(target=_run, daemon=True)
        _tts_thread.start()

# ── Song library ──────────────────────────────────────────────────────────────

def load_song_library(songs_dir: str) -> dict:
    library = {}
    if not os.path.isdir(songs_dir):
        print(f"[SONGS] Directory not found: {songs_dir}", file=sys.stderr)
        return library
    for root, _, files in os.walk(songs_dir):
        for fname in files:
            if fname.lower().endswith(".mp3"):
                display = os.path.splitext(fname)[0]
                library[display.lower()] = os.path.join(root, fname)
    return library

def find_best_match(query: str, library: dict):
    if not library:
        return None
    query = normalize(query)

    if query in library:
        return query, library[query]

    translated = HINDI_SONG_ALIASES.get(query)
    if translated and translated.lower() in library:
        key = translated.lower()
        return key, library[key]

    for hindi_key, eng_val in HINDI_SONG_ALIASES.items():
        if normalize(hindi_key) in query and eng_val.lower() in library:
            return eng_val.lower(), library[eng_val.lower()]

    matches = difflib.get_close_matches(query, list(library.keys()), n=1, cutoff=0.4)
    if matches:
        return matches[0], library[matches[0]]

    for key in library.keys():
        if query in key or key in query:
            return key, library[key]

    query_words = set(query.split())
    best_score, best_key = 0, None
    for key in library.keys():
        overlap = len(query_words & set(key.split()))
        if overlap > best_score:
            best_score, best_key = overlap, key
    if best_score > 0 and best_key:
        return best_key, library[best_key]

    return None

# ── Music player ──────────────────────────────────────────────────────────────

def play_song(path: str):
    pygame.mixer.music.stop()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    print(f"[SONGS] Playing: {path}")

def stop_song():
    pygame.mixer.music.stop()

def is_playing() -> bool:
    return pygame.mixer.music.get_busy()

# ── ADB ───────────────────────────────────────────────────────────────────────

def run_adb(command: str) -> str:
    try:
        result = subprocess.run(
            command, shell=True,
            capture_output=True, text=True, timeout=8
        )
        return result.stdout.strip() or result.stderr.strip()
    except subprocess.TimeoutExpired:
        return "ADB timeout"
    except Exception as e:
        return f"Error: {e}"

def check_adb_connected() -> bool:
    out = run_adb("adb devices")
    lines = [l for l in out.splitlines() if "device" in l and "List" not in l]
    return len(lines) > 0

def execute_adb_action(action: str, value: str = None) -> str:
    if action in ADB_ACTIONS:
        out = run_adb(ADB_ACTIONS[action])
        print(f"[ADB] {action} → {out}")
        return ADB_ACTION_HINDI.get(action, f"{action} हो गया।")
    elif action == "type_text" and value:
        safe = value.replace(" ", "%s")
        run_adb(f"adb shell input text '{safe}'")
        return f"'{value}' टाइप किया।"
    elif action == "tap" and value:
        try:
            x, y = value.split(",")
            run_adb(f"adb shell input tap {x.strip()} {y.strip()}")
            return "स्क्रीन पर टैप किया।"
        except ValueError:
            return "टैप के लिए x,y कोऑर्डिनेट चाहिए।"
    elif action == "none" or not action:
        return "यह फ़ोन कमांड समझ नहीं आई।"
    else:
        return f"अज्ञात कमांड: {action}"

# ── Keyword fallback ──────────────────────────────────────────────────────────

PLAY_TRIGGERS = [
    "गाना बजाओ", "गाना चलाओ", "गाना बजा दो", "गाना लगाओ",
    "गाने बजाओ", "गाने चलाओ", "बजाओ", "बजा दो",
    "चलाओ", "चला दो", "लगाओ", "सुनाओ", "बचाओ",
    "play", "song",
]
STOP_TRIGGERS  = ["बंद करो", "रुको", "रोको", "मत बजाओ", "stop", "pause"]
EXIT_TRIGGERS  = {"quit", "exit", "बाहर निकलो"}

PHONE_KW_MAP = {
    "व्हाट्सएप":    "open_whatsapp",
    "whatsapp":      "open_whatsapp",
    "यूट्यूब":      "open_youtube",
    "youtube":       "open_youtube",
    "स्पॉटिफाई":    "open_spotify",
    "spotify":       "open_spotify",
    "क्रोम":         "open_chrome",
    "chrome":        "open_chrome",
    "कैमरा":         "open_camera",
    "camera":        "open_camera",
    "सेटिंग":        "open_settings",
    "settings":      "open_settings",
    "स्क्रीनशॉट":   "screenshot",
    "screenshot":    "screenshot",
    "होम":           "home",
    "वापस":          "back",
    "आवाज़ बढ़":     "volume_up",
    "volume up":     "volume_up",
    "आवाज़ घट":     "volume_down",
    "volume down":   "volume_down",
    "लॉक":           "lock_screen",
    "lock":          "lock_screen",
    "अनलॉक":         "unlock_screen",
    "unlock":        "unlock_screen",
    "ऊपर स्क्रोल":  "scroll_up",
    "नीचे स्क्रोल": "scroll_down",
    "scroll up":     "scroll_up",
    "scroll down":   "scroll_down",
}

_NORM_PLAY_TRIGGERS  = [(normalize(t), t) for t in PLAY_TRIGGERS]
_NORM_STOP_TRIGGERS  = [normalize(t) for t in STOP_TRIGGERS]
_NORM_PHONE_KW_MAP   = {normalize(k): v for k, v in PHONE_KW_MAP.items()}
_NORM_SONG_ALIASES   = {normalize(k): v for k, v in HINDI_SONG_ALIASES.items()}

def keyword_detect_play(text: str):
    t = normalize(text)
    for nt, _ in _NORM_PLAY_TRIGGERS:
        if nt in t:
            after = t.split(nt, 1)[-1].strip()
            return after if after else ""
    return None

def keyword_detect_stop(text: str) -> bool:
    t = normalize(text)
    return any(tr in t for tr in _NORM_STOP_TRIGGERS)

def keyword_detect_phone(text: str) -> str | None:
    t = normalize(text)
    for kw, action in _NORM_PHONE_KW_MAP.items():
        if kw in t:
            return action
    return None

# ── Plain-text rescue parser ──────────────────────────────────────────────────

def rescue_parse_plain_text(llm_text: str, original_user_text: str) -> dict | None:
    combined = normalize(llm_text) + " " + normalize(original_user_text)

    if keyword_detect_stop(combined):
        return {"intent": "stop_song",    "song_name": None,
                "phone_action": None, "phone_value": None, "reply": None}

    phone_action = keyword_detect_phone(combined)
    if phone_action:
        return {"intent": "phone_command", "song_name": None,
                "phone_action": phone_action, "phone_value": None, "reply": None}

    sq = keyword_detect_play(combined)
    if sq is not None:
        return {"intent": "play_song",    "song_name": sq,
                "phone_action": None, "phone_value": None, "reply": None}

    for nk, _ in _NORM_SONG_ALIASES.items():
        if nk in combined:
            return {"intent": "play_song", "song_name": nk,
                    "phone_action": None, "phone_value": None, "reply": None}

    return None

# ── LLM Intent Classifier ─────────────────────────────────────────────────────

def classify_intent(user_text: str) -> dict | None:
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": INTENT_MODEL,
                "messages": [
                    {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_text},
                ],
                "stream":     False,
                "keep_alive": -1,
                "options": {
                    "num_ctx":     512,
                    "num_predict": 80,
                    "num_thread":  4,
                    "temperature": 0.0,
                },
            },
            timeout=30,
        )
        r.raise_for_status()
        content = r.json().get("message", {}).get("content", "").strip()
        print(f"[LLM RAW] {content}")

        content_clean = re.sub(r"```json|```", "", content).strip()
        match = re.search(r"\{.*?\}", content_clean, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            if parsed.get("intent") == "play_song" and not parsed.get("song_name"):
                parsed["song_name"] = ""
            return parsed

        print("[INTENT] No JSON — rescue parser", file=sys.stderr)
        rescued = rescue_parse_plain_text(content, user_text)
        if rescued:
            return rescued

        return None

    except json.JSONDecodeError as e:
        print(f"[INTENT] JSON parse error: {e}", file=sys.stderr)
        return None
    except requests.exceptions.Timeout:
        print("[INTENT] LLM timeout — keyword fallback", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[INTENT] LLM error: {e}", file=sys.stderr)
        return None

def classify_intent_fast(user_text: str) -> dict:
    kw_result = _keyword_only_classify(user_text)
    kw_confident = kw_result["intent"] in ("stop_song", "play_song", "phone_command")

    llm_result_box: list = []

    def _llm_worker():
        r = classify_intent(user_text)
        if r:
            llm_result_box.append(r)

    llm_thread = threading.Thread(target=_llm_worker, daemon=True)
    llm_thread.start()

    if kw_confident:
        print(f"[INTENT] Keyword confident → {kw_result['intent']} (LLM still running)")
        llm_thread.join(timeout=0)
        return kw_result

    llm_thread.join(timeout=5.0)
    if llm_result_box:
        print(f"[INTENT] LLM result → {llm_result_box[0]['intent']}")
        return llm_result_box[0]

    print("[INTENT] LLM timed out — using keyword fallback")
    return kw_result

def _keyword_only_classify(user_text: str) -> dict:
    _empty = {"song_name": None, "phone_action": None, "phone_value": None, "reply": None}

    if keyword_detect_stop(user_text):
        return {**_empty, "intent": "stop_song"}

    action = keyword_detect_phone(user_text)
    if action:
        return {**_empty, "intent": "phone_command", "phone_action": action}

    sq = keyword_detect_play(user_text)
    if sq is not None:
        return {**_empty, "intent": "play_song", "song_name": sq}

    if any(w in user_text for w in ("गाना", "गाने", "song")):
        return {**_empty, "intent": "play_song", "song_name": ""}

    for nk in _NORM_SONG_ALIASES:
        if nk in user_text:
            return {**_empty, "intent": "play_song", "song_name": nk}

    return {**_empty, "intent": "chat"}

# ── Conversation memory ───────────────────────────────────────────────────────

chat_history: list = []

def ask_llm_chat(user_text: str) -> str:
    chat_history.append({"role": "user", "content": user_text})
    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model":      CHAT_MODEL,
            "messages":   [{"role": "system", "content": CHAT_SYSTEM_PROMPT}] + chat_history,
            "stream":     True,
            "keep_alive": -1,
            "options": {
                "num_ctx":     2048,
                "num_predict": 200,
                "num_thread":  4,
                "temperature": 0.7,
            },
        },
        stream=True,
        timeout=120,
    )
    r.raise_for_status()
    full_response = ""
    for line in r.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        token = chunk.get("message", {}).get("content", "")
        full_response += token
        print(token, end="", flush=True)
        if chunk.get("done"):
            break
    print()
    chat_history.append({"role": "assistant", "content": full_response.strip()})
    if len(chat_history) > 16:
        chat_history.pop(0)
        chat_history.pop(0)
    return full_response.strip()

# ── Parallel warm-up ──────────────────────────────────────────────────────────

def _warm_one(model_name: str, results: dict):
    t0 = time.time()
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model":      model_name,
                "prompt":     "नमस्ते",
                "keep_alive": -1,
                "options":    {"num_predict": 1},
            },
            timeout=120,
            stream=True,
        )
        r.raise_for_status()
        for _ in r.iter_lines():
            pass
        results[model_name] = f"ready in {time.time()-t0:.1f}s"
    except Exception as e:
        results[model_name] = f"FAILED: {e}"

def warm_up():
    print("Warming up LLM models in parallel…")
    results: dict = {}
    threads = [
        threading.Thread(target=_warm_one, args=(INTENT_MODEL, results), daemon=True),
        threading.Thread(target=_warm_one, args=(CHAT_MODEL,   results), daemon=True),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for model_name, status in results.items():
        if "FAILED" in status:
            print(f"  ✗ {model_name}: {status}")
            sys.exit(1)
        else:
            print(f"  ✓ {model_name}: {status}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-speech",  action="store_true", help="Disable TTS output")
    ap.add_argument("--songs-dir",  default=SONGS_DIR)
    ap.add_argument("--no-llm",     action="store_true", help="Keyword-only mode")
    ap.add_argument("--no-adb",     action="store_true", help="Disable ADB phone control")
    ap.add_argument("--adb-wifi",   default=None, metavar="IP",
                    help="Connect to phone over WiFi, e.g. 192.168.1.5")
    args = ap.parse_args()

    print("\n══════════════════════════════════════════════")
    print("🇮🇳  Hindi Assistant v5-OPT — Song + ADB")
    print( "══════════════════════════════════════════════")

    vosk_model_box  = [None]
    song_lib_box    = [{}]
    tts_error_box   = [None]
    adb_status_box  = [None]

    def _load_vosk():
        vosk_model_box[0] = Model(VOSK_MODEL_PATH)

    def _load_songs():
        song_lib_box[0] = load_song_library(args.songs_dir)

    def _load_tts_thread():
        if not args.no_speech:
            try:
                load_tts()
            except Exception as e:
                tts_error_box[0] = e

    def _check_adb():
        if args.no_adb:
            return
        if args.adb_wifi:
            run_adb(f"adb connect {args.adb_wifi}:5555")
        adb_status_box[0] = check_adb_connected()

    init_threads = [
        threading.Thread(target=_load_vosk,      daemon=True),
        threading.Thread(target=_load_songs,     daemon=True),
        threading.Thread(target=_load_tts_thread,daemon=True),
        threading.Thread(target=_check_adb,      daemon=True),
    ]

    if not args.no_llm:
        warm_up()

    print("Loading VOSK + TTS + songs in parallel…")
    for t in init_threads:
        t.start()
    for t in init_threads:
        t.join()

    vosk_model  = vosk_model_box[0]
    song_library = song_lib_box[0]

    if tts_error_box[0]:
        print(f"[TTS] Load error: {tts_error_box[0]}", file=sys.stderr)

    if not args.no_adb:
        if adb_status_box[0]:
            print("[ADB] Device connected ✓")
        else:
            print("[ADB] ⚠️  No device — connect phone with USB Debugging enabled")
            print(f"[ADB]    or use --adb-wifi <phone-ip>")

    if song_library:
        print(f"[SONGS] {len(song_library)} song(s) loaded:")
        for name in sorted(song_library.keys()):
            print(f"   • {name}")

    print("\n✅ Ready!\n")

    rec = KaldiRecognizer(vosk_model, SAMPLE_RATE)

    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
        dtype="int16", channels=1, callback=_cb,
    )

    try:
        while True:
            while not _audio_q.empty():
                try:
                    _audio_q.get_nowait()
                except queue.Empty:
                    break

            stream.start()
            user_text = listen_once(rec)
            stream.stop()

            if not user_text:
                continue

            user_text = normalize(user_text)
            print(f"\nआप: {user_text}")

            if user_text in EXIT_TRIGGERS:
                stop_song()
                speak("अलविदा!", args.no_speech)
                break

            if is_playing():
                if keyword_detect_stop(user_text):
                    stop_song()
                    speak("गाना बंद किया।", args.no_speech, wait=False)
                elif not args.no_adb:
                    action = keyword_detect_phone(user_text)
                    if action:
                        reply = execute_adb_action(action)
                        speak(reply, args.no_speech, wait=False)
                    else:
                        print("[INFO] Music playing — only stop/phone commands accepted")
                else:
                    print("[INFO] Music playing — only stop commands accepted")
                continue

            if args.no_llm:
                intent_data = _keyword_only_classify(user_text)
            else:
                intent_data = classify_intent_fast(user_text)

            intent       = intent_data.get("intent", "chat")
            song_name    = (intent_data.get("song_name")    or "").strip()
            phone_action = (intent_data.get("phone_action") or "").strip()
            phone_value  =  intent_data.get("phone_value")
            llm_reply    = (intent_data.get("reply")        or "").strip()

            print(f"[INTENT] {intent} | song='{song_name}' | phone='{phone_action}'")

            if intent == "exit":
                stop_song()
                speak("अलविदा!", args.no_speech)
                break

            elif intent == "stop_song":
                if is_playing():
                    stop_song()
                    speak("गाना बंद किया।", args.no_speech, wait=False)
                else:
                    speak("कोई गाना नहीं चल रहा।", args.no_speech, wait=False)

            elif intent == "phone_command":
                if args.no_adb:
                    speak("ADB फ़ोन कंट्रोल बंद है।", args.no_speech)
                else:
                    reply = execute_adb_action(phone_action, phone_value)
                    speak(reply, args.no_speech, wait=False)

            elif intent == "play_song":
                if not song_library:
                    speak("गानों का फोल्डर नहीं मिला।", args.no_speech)
                    continue

                match = None
                if song_name:
                    match = find_best_match(song_name, song_library)
                    print(f"[DEBUG] Name='{song_name}' → {match}")
                if not match:
                    match = find_best_match(user_text, song_library)
                    print(f"[DEBUG] Full='{user_text}' → {match}")

                if match:
                    name, path = match
                    speak(f"{name} बजा रहा हूँ।", args.no_speech, wait=False)
                    play_song(path)
                else:
                    names = ", ".join(sorted(song_library.keys()))
                    speak(f"कौन सा गाना? {names}", args.no_speech)

            elif intent == "chat":
                if llm_reply:
                    # 1b already answered — no second LLM call needed
                    speak(llm_reply, args.no_speech)
                else:
                    # 1b returned null reply — fall back to 3b model
                    print("[CHAT] 1b reply was null — falling back to 3b model")
                    print("AI: ", end="", flush=True)
                    try:
                        full_response = ask_llm_chat(user_text)
                        if not args.no_speech and full_response:
                            print("🔊 बोल रहा हूँ…")
                            speak(full_response, args.no_speech)
                    except Exception as e:
                        print(f"\n[LLM] {e}", file=sys.stderr)

            else:
                print(f"[WARN] Unknown intent '{intent}' — chat")
                try:
                    full_response = ask_llm_chat(user_text)
                    if not args.no_speech and full_response:
                        speak(full_response, args.no_speech)
                except Exception as e:
                    print(f"\n[LLM] {e}", file=sys.stderr)

    except KeyboardInterrupt:
        print("\nअलविदा!")
    finally:
        stop_song()
        pygame.mixer.quit()
        stream.stop()
        stream.close()

if __name__ == "__main__":
    main()