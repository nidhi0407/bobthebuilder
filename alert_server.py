"""
BobTheBuilder - Construction Safety Alert Server
Hackathon Demo Backend with ElevenLabs Text-to-Speech
"""

import os
import time
import math
import struct
import wave
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from dotenv import load_dotenv
import requests

# Load environment variables from .env if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("AlertServer")

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(BASE_DIR))
CORS(app)

# Default ElevenLabs Voice IDs
# Adam (deep, authoritative voice): pNInz6obpgDQGcFmaJgB
# Rachel (crisp, professional voice): 21m00Tcm4TlvDq8ikWAM
DEFAULT_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "pNInz6obpgDQGcFmaJgB")

# Shared In-Memory Alert State
alert_state = {
    "active": False,
    "worker_message": "All clear. Normal site operations.",
    "forklift_message": "All clear. Travel lanes unobstructed.",
    "timestamp": time.time()
}

def generate_fallback_audio(output_path: Path, tone_type: str = "warning"):
    """
    Generates a clean PCM WAV audio alert file using Python standard library.
    Provides audible emergency feedback if ElevenLabs key is not set.
    """
    try:
        sample_rate = 24000
        duration = 1.4  # seconds
        num_samples = int(sample_rate * duration)
        
        raw_samples = []
        for i in range(num_samples):
            t = i / sample_rate
            cycle = t % 0.35
            if cycle < 0.22:
                freq = 880.0 if tone_type == "worker" else 720.0
                sample = 0.6 * math.sin(2.0 * math.pi * freq * t)
                sample += 0.3 * math.sin(2.0 * math.pi * (freq * 1.5) * t)
            else:
                sample = 0.0
            
            sample_val = int(max(-1.0, min(1.0, sample)) * 32767)
            raw_samples.append(sample_val)
        
        with wave.open(str(output_path), "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            packed = struct.pack(f"<{len(raw_samples)}h", *raw_samples)
            wav_file.writeframes(packed)
            
        logger.info(f"Generated clean synthetic fallback audio at {output_path.name}")
    except Exception as e:
        logger.error(f"Failed to generate fallback audio: {e}")


def synthesize_elevenlabs_speech(text: str, output_path: Path, role: str = "worker"):
    """
    Synthesizes speech using ElevenLabs API.
    If API key is missing or request fails, falls back gracefully to synthetic tone.
    """
    api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    
    if not api_key:
        logger.warning(
            f"ELEVENLABS_API_KEY is not set. Generating fallback audio for {role}. "
            "Add ELEVENLABS_API_KEY to your .env file to enable realistic AI voice."
        )
        generate_fallback_audio(output_path, tone_type=role)
        return False
    
    voice_id = os.environ.get(
        f"ELEVENLABS_{role.upper()}_VOICE_ID",
        DEFAULT_VOICE_ID
    )
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.45,
            "similarity_boost": 0.85,
            "style": 0.2,
            "use_speaker_boost": True
        }
    }
    
    try:
        logger.info(f"Calling ElevenLabs TTS for {role}: '{text}' (voice={voice_id})")
        response = requests.post(url, json=payload, headers=headers, timeout=12)
        
        if response.status_code == 200 and len(response.content) > 100:
            with open(output_path, "wb") as f:
                f.write(response.content)
            logger.info(f"ElevenLabs audio saved to {output_path.name} ({len(response.content)} bytes)")
            return True
        else:
            logger.error(f"ElevenLabs TTS returned [{response.status_code}]: {response.text}")
            generate_fallback_audio(output_path, tone_type=role)
            return False
    except Exception as exc:
        logger.error(f"Exception connecting to ElevenLabs API: {exc}")
        generate_fallback_audio(output_path, tone_type=role)
        return False


# Prepare default audio files
worker_audio_path = AUDIO_DIR / "worker.mp3"
forklift_audio_path = AUDIO_DIR / "forklift.mp3"
if not worker_audio_path.exists():
    generate_fallback_audio(worker_audio_path, "worker")
if not forklift_audio_path.exists():
    generate_fallback_audio(forklift_audio_path, "forklift")


# --- HTTP Routes ---

@app.route("/status", methods=["GET"])
def get_status():
    """Returns the current alert state."""
    return jsonify(alert_state), 200


@app.route("/trigger", methods=["POST"])
def trigger_alert():
    """
    Triggers an active alert.
    Accepts JSON: { "worker_message": "...", "forklift_message": "..." }
    """
    data = request.get_json(silent=True) or {}
    worker_msg = data.get("worker_message", "Warning worker! Moving forklift detected approaching your quadrant!")
    forklift_msg = data.get("forklift_message", "Emergency stop! Worker detected in vehicle travel corridor!")

    logger.warning("🚨 EMERGENCY ALERT TRIGGERED")
    logger.warning(f"   Worker:   {worker_msg}")
    logger.warning(f"   Forklift: {forklift_msg}")

    # Synthesize audio files
    synthesize_elevenlabs_speech(worker_msg, worker_audio_path, role="worker")
    synthesize_elevenlabs_speech(forklift_msg, forklift_audio_path, role="forklift")

    # Update in-memory state
    alert_state["active"] = True
    alert_state["worker_message"] = worker_msg
    alert_state["forklift_message"] = forklift_msg
    alert_state["timestamp"] = time.time()

    return jsonify({
        "status": "alert_triggered",
        "state": alert_state
    }), 200


@app.route("/clear", methods=["POST"])
def clear_alert():
    """Resets the alert state to All Clear."""
    logger.info("🟢 ALERT CLEARED -> ALL CLEAR")
    alert_state["active"] = False
    alert_state["worker_message"] = "All clear. Hazard zone resolved."
    alert_state["forklift_message"] = "All clear. Travel lanes unobstructed."
    alert_state["timestamp"] = time.time()

    return jsonify({
        "status": "alert_cleared",
        "state": alert_state
    }), 200


@app.route("/worker", methods=["GET"])
def serve_worker_page():
    """Serves the phone-facing worker page."""
    return send_from_directory(str(BASE_DIR), "worker.html")


@app.route("/forklift", methods=["GET"])
def serve_forklift_page():
    """Serves the phone-facing forklift operator page."""
    return send_from_directory(str(BASE_DIR), "forklift.html")


@app.route("/audio/<filename>", methods=["GET"])
def serve_audio(filename):
    """
    Serves audio files with proper auto-detected MIME type (WAV vs MP3)
    and aggressive cache-busting headers so mobile browsers always load fresh audio.
    """
    target_file = AUDIO_DIR / filename
    # Allow looking up worker.mp3 even if worker.wav requested, or vice versa
    if not target_file.exists():
        alt_name = filename.rsplit(".", 1)[0] + (".wav" if filename.endswith(".mp3") else ".mp3")
        target_file = AUDIO_DIR / alt_name
        if not target_file.exists():
            return jsonify({"error": "Audio file not found"}), 404

    with open(target_file, "rb") as f:
        content = f.read()

    # Detect actual file content signature
    if content.startswith(b"RIFF"):
        mime_type = "audio/wav"
    else:
        mime_type = "audio/mpeg"

    response = Response(content, mimetype=mime_type)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["Accept-Ranges"] = "bytes"
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.route("/", methods=["GET"])
def index():
    """Dashboard homepage with live controls."""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BobTheBuilder Safety Server</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0f172a;
                color: #f8fafc;
                margin: 0;
                padding: 24px;
                display: flex;
                flex-direction: column;
                align-items: center;
            }}
            .card {{
                max-width: 600px;
                width: 100%;
                background: #1e293b;
                border-radius: 16px;
                padding: 24px;
                box-shadow: 0 10px 25px rgba(0,0,0,0.5);
                border: 1px solid #334155;
            }}
            h1 {{ margin-top: 0; color: #f59e0b; display: flex; align-items: center; gap: 8px; }}
            .btn-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 20px 0; }}
            .btn {{
                display: block;
                text-align: center;
                padding: 16px;
                border-radius: 12px;
                text-decoration: none;
                font-weight: bold;
                font-size: 1.1rem;
            }}
            .btn-worker {{ background: #2563eb; color: white; }}
            .btn-forklift {{ background: #d97706; color: white; }}
            .btn-trigger {{ background: #dc2626; color: white; border: none; cursor: pointer; padding: 16px; font-size: 1.1rem; border-radius: 8px; font-weight: bold; width: 100%; }}
            .btn-clear {{ background: #16a34a; color: white; border: none; cursor: pointer; padding: 14px; font-size: 1rem; border-radius: 8px; font-weight: bold; width: 100%; margin-top: 10px; }}
            pre {{ background: #020617; padding: 12px; border-radius: 8px; font-size: 0.85rem; overflow-x: auto; color: #38bdf8; }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>🚧 BobTheBuilder Safety Center</h1>
            <p>Phone Client Links:</p>
            <div class="btn-grid">
                <a href="/worker" class="btn btn-worker" target="_blank">👷 Worker Screen</a>
                <a href="/forklift" class="btn btn-forklift" target="_blank">🚜 Forklift Screen</a>
            </div>
            <h3>Quick Actions</h3>
            <button class="btn-trigger" onclick="triggerDemoAlert()">🚨 Fire Emergency Alert</button>
            <button class="btn-clear" onclick="clearAlert()">🟢 All Clear</button>
            <h3>Live Status</h3>
            <pre id="status-view">Loading...</pre>
        </div>
        <script>
            async function refreshStatus() {{
                try {{
                    const res = await fetch('/status');
                    const data = await res.json();
                    document.getElementById('status-view').textContent = JSON.stringify(data, null, 2);
                }} catch(e) {{}}
            }}
            async function triggerDemoAlert() {{
                await fetch('/trigger', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{
                        worker_message: "Warning worker! Forklift approaching blind corner on your right!",
                        forklift_message: "Emergency stop! Worker stepping into path 5 meters ahead!"
                    }})
                }});
                refreshStatus();
            }}
            async function clearAlert() {{
                await fetch('/clear', {{ method: 'POST' }});
                refreshStatus();
            }}
            refreshStatus();
            setInterval(refreshStatus, 2000);
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting BobTheBuilder Alert Server on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
