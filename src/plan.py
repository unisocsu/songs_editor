#!/usr/bin/env python3
"""שלב 2: תכנון העיבוד.

קלט:  ההקלטה המקורית + out/analysis.json (מהשלב הקודם)
פלט:  out/plan.json  - סגנון, קצב, כלים, ליווי והרמוניה (נבנית מהלחן, לא מזוהה מההקלטה)

הסגנון:
  --style "טקסט"   נקבע ידנית (למשל מ-workflow_dispatch)
  בלי --style      Gemini מאזין להקלטה ומזהה את הסגנון בעצמו

דורש משתנה סביבה GEMINI_API_KEY. הקול המקורי לא משתנה - התוכנית מתארת רק את הליווי.
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def log(msg):
    print(f"[plan] {msg}", flush=True)


# ------------------------------------------------------------ מאפייני קצב מקומיים
def rhythm_features(audio: Path):
    """האם יש פעימה יציבה בהקלטה? (0 = חופשי, קרוב ל-1 = פעימה ברורה)"""
    import librosa

    y, sr = librosa.load(str(audio), sr=22050, mono=True)
    hop = 512
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    env = env - env.mean()
    ac = librosa.autocorrelate(env)
    ac = ac / (ac[0] + 1e-9)
    fps = sr / hop
    lo, hi = int(fps * 60 / 200), int(fps * 60 / 40)  # 40..200 BPM
    seg = ac[lo:hi]
    lag = lo + int(np.argmax(seg))
    return {
        "beat_regularity": round(float(seg.max()), 3),
        "candidate_bpm": round(60.0 * fps / lag, 1),
    }


# --------------------------------------------------------------------- Gemini
def to_small_mp3(src: Path) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        dst = Path(d) / "a.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "22050",
             "-b:a", "64k", str(dst)],
            check=True,
        )
        return dst.read_bytes()


def call_gemini(model, api_key, prompt, audio_bytes=None, retries=3):
    parts = [{"text": prompt}]
    if audio_bytes:
        parts.append({"inline_data": {"mime_type": "audio/mp3",
                                      "data": base64.b64encode(audio_bytes).decode()}})
    body = json.dumps({
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4},
    }).encode()
    req = urllib.request.Request(
        API.format(model=model), data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.load(r)
            text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
            return text
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
            if e.code in (429, 500, 503):
                continue
            break
        except Exception as e:  # noqa: BLE001
            last = str(e)
    raise RuntimeError(f"Gemini failed: {last}")


def parse_json(text):
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):]
    return json.loads(t)


# ---------------------------------------------------------------------- פרומפט
SCHEMA = """{
  "style": {"name": "string", "description": "string", "detected_from_audio": true},
  "rhythm": {"feel": "steady | free | mixed", "bpm": 0, "time_signature": "4/4"},
  "instruments": [{"role": "drums | bass | harmony | countermelody | pad", "gm_program": 0, "notes": "string"}],
  "drums": {"enabled": true, "pattern": "short text description"},
  "sections": [{"start": 0.0, "end": 0.0, "intensity": 0.0,
                "chords": [{"start": 0.0, "end": 0.0, "chord": "Dm"}]}],
  "warnings": ["string"]
}"""


def build_prompt(analysis, rhythm, style):
    notes = " ".join(f"{n['pitch']}:{n['start']}:{round(n['end'] - n['start'], 2)}"
                     for n in analysis["notes"])
    if style:
        style_part = (f'The requested arrangement style is: "{style}". '
                      'Set detected_from_audio=false and base the plan on this style.')
    else:
        style_part = ("No style was requested. LISTEN to the attached recording and identify its musical "
                      "style/genre/tradition yourself, then plan an arrangement that fits it. "
                      "Set detected_from_audio=true.")
    return f"""You are an arranger. A person recorded a melody a cappella (no accompaniment). You must plan the
ACCOMPANIMENT only (drums, bass, harmony instruments). The sung melody itself will NOT be changed or re-sung.

{style_part}

Facts about the recording:
- duration_sec: {analysis['duration_sec']}
- beat_regularity (0 free rhythm .. 1 very steady): {rhythm['beat_regularity']}, candidate_bpm: {rhythm['candidate_bpm']}
- pitch classes used (seconds): {json.dumps(analysis['melody'].get('pitch_class_seconds', {}), ensure_ascii=False)}
- melody notes as midi_pitch:start_sec:duration_sec: {notes}

Rules:
1. Chords/harmony must be derived from the melody notes above so they support it. If the melody is modal or
   uses many non-diatonic notes, prefer a drone/pedal or sparse harmony and add a warning.
2. If rhythm is free (low beat_regularity and you hear no steady pulse), set rhythm.feel="free", bpm=null,
   drums.enabled=false, and prefer sustained pads/strings/bass drone. Otherwise give a bpm.
3. Chord names use simple forms: C, Dm, G7, Am, Bb, F#m7b5. Times are seconds from the start of the recording
   and must stay within duration_sec. Sections must cover the recording without overlap.
4. instruments[].gm_program is a General MIDI program number (0-127). Drums use channel 10 (program ignored).
5. Keep it playable: at most 6 instruments.
6. Output ONLY valid JSON matching this shape (no commentary):
{SCHEMA}"""


# -------------------------------------------------------------------- אימות
def validate(plan, duration):
    for k in ("style", "rhythm", "instruments", "sections"):
        if k not in plan:
            raise ValueError(f"missing key: {k}")
    plan.setdefault("drums", {"enabled": False, "pattern": ""})
    plan.setdefault("warnings", [])
    plan["instruments"] = plan["instruments"][:6]
    for ins in plan["instruments"]:
        ins["gm_program"] = int(min(max(int(ins.get("gm_program", 0)), 0), 127))
    for s in plan["sections"]:
        s["start"] = float(max(0.0, min(float(s["start"]), duration)))
        s["end"] = float(max(s["start"], min(float(s["end"]), duration)))
        for c in s.get("chords", []):
            c["start"] = float(max(0.0, min(float(c["start"]), duration)))
            c["end"] = float(max(c["start"], min(float(c["end"]), duration)))
    if plan["rhythm"].get("feel") == "free":
        plan["drums"]["enabled"] = False
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="ההקלטה המקורית")
    ap.add_argument("--analysis", default="out/analysis.json")
    ap.add_argument("--out", default="out/plan.json")
    ap.add_argument("--style", default="", help="סגנון ידני; ריק = זיהוי אוטומטי מההקלטה")
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    ap.add_argument("--dry-run", action="store_true", help="הדפס את הפרומפט בלי לקרוא ל-API")
    args = ap.parse_args()

    analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    rhythm = rhythm_features(Path(args.audio))
    log(f"מאפייני קצב: {rhythm}")
    style = args.style.strip()
    prompt = build_prompt(analysis, rhythm, style)

    if args.dry_run:
        print(prompt[:3000])
        return

    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        sys.exit("חסר GEMINI_API_KEY")

    audio = to_small_mp3(Path(args.audio))
    log(f"שולח ל-{args.model} (סגנון: {style or 'זיהוי אוטומטי'})...")
    plan = None
    for attempt in range(2):
        text = call_gemini(args.model, key, prompt, audio)
        try:
            plan = validate(parse_json(text), analysis["duration_sec"])
            break
        except Exception as e:  # noqa: BLE001
            log(f"תשובה לא תקינה ({e}), מנסה שוב...")
    if plan is None:
        sys.exit("המודל לא החזיר תוכנית תקינה")

    plan["rhythm_measured"] = rhythm
    plan["model"] = args.model
    Path(args.out).write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"נשמר {args.out}: סגנון = {plan['style'].get('name')}")


if __name__ == "__main__":
    main()
