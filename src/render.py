#!/usr/bin/env python3
"""שלב 3: ביצוע העיבוד.

קלט:  plan.json (משלב התכנון) + ההקלטה המקורית (+ melody.mid למשחק כפול של הלחן)
פלט:  out/arrangement.mid  - הליווי כ-MIDI
      out/result.mp3        - ההקלטה המקורית שלך + הליווי, מעורבבים
      out/accompaniment.mp3 - הליווי בלבד

הקול המקורי לא משתנה בכלל. זמני האקורדים בתוכנית הם שניות מתחילת ההקלטה,
ולכן הליווי מתיישר אוטומטית גם כשהקצב חופשי.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

SR = 44100
SOUNDFONT_CANDIDATES = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
]
NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
QUALITIES = [  # (prefix, intervals) - הארוך קודם
    ("m7b5", [0, 3, 6, 10]), ("dim7", [0, 3, 6, 9]), ("dim", [0, 3, 6]), ("aug", [0, 4, 8]),
    ("maj7", [0, 4, 7, 11]), ("M7", [0, 4, 7, 11]), ("m7", [0, 3, 7, 10]), ("min7", [0, 3, 7, 10]),
    ("min", [0, 3, 7]), ("sus4", [0, 5, 7]), ("sus2", [0, 2, 7]), ("m6", [0, 3, 7, 9]),
    ("m", [0, 3, 7]), ("7", [0, 4, 7, 10]), ("6", [0, 4, 7, 9]), ("", [0, 4, 7]),
]


def log(msg):
    print(f"[render] {msg}", flush=True)


def parse_chord(name):
    """'Dm7' -> (root_pc, intervals, bass_pc). None אם לא ניתן לפענח."""
    if not name:
        return None
    name = name.strip()
    if name.upper() in ("N", "N.C.", "NC", "-"):
        return None
    main, _, slash = name.partition("/")
    m = re.match(r"^([A-Ga-g])([#b♯♭]?)(.*)$", main)
    if not m:
        return None
    root = NOTE_PC[m.group(1).upper()]
    acc = m.group(2)
    root += 1 if acc in ("#", "♯") else -1 if acc in ("b", "♭") else 0
    rest = m.group(3).replace("add9", "").replace("add", "")
    intervals = None
    for prefix, iv in QUALITIES:
        if rest.startswith(prefix):
            intervals = iv
            break
    if intervals is None:
        intervals = [0, 3, 7] if rest.startswith("m") and not rest.startswith("maj") else [0, 4, 7]
    bass = None
    if slash:
        sm = re.match(r"^([A-Ga-g])([#b♯♭]?)", slash)
        if sm:
            bass = NOTE_PC[sm.group(1).upper()] + (1 if sm.group(2) in ("#", "♯") else -1 if sm.group(2) in ("b", "♭") else 0)
    return root % 12, intervals, (bass % 12 if bass is not None else None)


def voicing(root_pc, intervals, base_octave_c):
    """אקורד צמוד שמתחיל בשורש (base_octave_c = מספר MIDI של C באוקטבה הרצויה)."""
    root = base_octave_c + root_pc
    if root >= base_octave_c + 7:  # שומר על טווח אחיד
        root -= 12
    return [root + i for i in intervals]


def section_intensity(sections, t):
    for s in sections:
        if s["start"] <= t < s["end"]:
            return float(s.get("intensity", 0.5))
    return 0.5


def build_midi(plan, melody_path, duration):
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    sections = plan.get("sections", [])
    chords = []
    for s in sections:
        for c in s.get("chords", []):
            parsed = parse_chord(c.get("chord", ""))
            if parsed and c["end"] - c["start"] > 0.05:
                chords.append((float(c["start"]), float(c["end"]), parsed))
    chords.sort()
    if not chords:
        log("אזהרה: אין אקורדים בתוכנית - הליווי יהיה ריק")

    rhythm = plan.get("rhythm", {})
    bpm = rhythm.get("bpm")
    drums_on = bool(plan.get("drums", {}).get("enabled")) and rhythm.get("feel") != "free" and bpm
    instruments = plan.get("instruments", [])
    n_made = 0

    for ins in instruments:
        role = str(ins.get("role", "")).lower()
        prog = int(ins.get("gm_program", 0))
        if role == "drums":
            if not drums_on:
                continue
            drum = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
            beat = 60.0 / float(bpm)
            t, k = 0.0, 0
            while t < duration:
                v = int(50 + 40 * section_intensity(sections, t))
                if k % 4 in (0, 2):
                    drum.notes.append(pretty_midi.Note(v + 10, 36, t, t + 0.1))
                if k % 4 in (1, 3):
                    drum.notes.append(pretty_midi.Note(v, 38, t, t + 0.1))
                drum.notes.append(pretty_midi.Note(max(30, v - 25), 42, t, t + 0.05))
                drum.notes.append(pretty_midi.Note(max(30, v - 30), 42, t + beat / 2, t + beat / 2 + 0.05))
                t += beat
                k += 1
            pm.instruments.append(drum)
            n_made += 1
            continue

        inst = pretty_midi.Instrument(program=prog, name=role or "inst")
        if role == "bass":
            for start, end, (root, iv, bass) in chords:
                p = 36 + ((bass if bass is not None else root) % 12)
                if p > 47:
                    p -= 12
                vel = int(55 + 40 * section_intensity(sections, start))
                inst.notes.append(pretty_midi.Note(vel, p, start, max(start + 0.2, end - 0.03)))
        elif role in ("harmony", "pad"):
            octave_c = 60 if role == "pad" else 48  # פד גבוה יותר, הרמוניה נמוכה יותר
            for start, end, (root, iv, bass) in chords:
                vel = int((30 if role == "pad" else 40) + 40 * section_intensity(sections, start))
                for p in voicing(root, iv, octave_c):
                    inst.notes.append(pretty_midi.Note(min(vel, 110), int(p), start + 0.02, max(start + 0.2, end - 0.02)))
        elif role == "countermelody":
            # הכפלת הלחן המקורי (כולל Pitch Bend) בכלי נוסף, בעוצמה נמוכה
            if melody_path and Path(melody_path).exists():
                src = pretty_midi.PrettyMIDI(str(melody_path)).instruments[0]
                for n in src.notes:
                    inst.notes.append(pretty_midi.Note(max(25, int(n.velocity * 0.6)), n.pitch, n.start, n.end))
                inst.pitch_bends.extend(src.pitch_bends)
        else:
            log(f"תפקיד לא מוכר '{role}' - מדלג")
            continue
        if inst.notes:
            pm.instruments.append(inst)
            n_made += 1
    log(f"נוצרו {n_made} ערוצי ליווי, {len(chords)} אקורדים, תופים: {'כן' if drums_on else 'לא'}")
    return pm


def find_soundfont(arg):
    if arg and Path(arg).exists():
        return arg
    for c in SOUNDFONT_CANDIDATES:
        if Path(c).exists():
            return c
    return None


def synthesize(pm, midi_path: Path, wav_path: Path, soundfont):
    if shutil.which("fluidsynth") and soundfont:
        log(f"מסנתז עם fluidsynth ({Path(soundfont).name})...")
        subprocess.run(["fluidsynth", "-ni", "-q", "-g", "0.8", "-r", str(SR), "-F", str(wav_path),
                        soundfont, str(midi_path)], check=True)
        return
    log("אזהרה: fluidsynth/SoundFont לא נמצאו - סינתזה בסיסית (גל סינוס), איכות נמוכה")
    audio = pm.synthesize(fs=SR)
    sf.write(str(wav_path), np.stack([audio, audio], axis=1), SR)


def load_stereo(path):
    y, sr = sf.read(str(path), always_2d=True)
    assert sr == SR
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    return y[:, :2].astype(np.float64)


def rms_active(y, thr=0.01):
    m = np.abs(y).max(axis=1)
    act = y[m > thr]
    return float(np.sqrt(np.mean(act ** 2))) if len(act) else 1e-6


def mix(voice, acc, gain):
    n = max(len(voice), len(acc))
    voice = np.pad(voice, ((0, n - len(voice)), (0, 0)))
    acc = np.pad(acc, ((0, n - len(acc)), (0, 0)))
    acc = acc * (gain * rms_active(voice) / rms_active(acc))
    out = voice + acc
    peak = np.abs(out).max()
    if peak > 0.95:
        out = out * (0.95 / peak)
    return out


def to_mp3(wav_path: Path, mp3_path: Path):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path), "-b:a", "192k", str(mp3_path)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="ההקלטה המקורית")
    ap.add_argument("--plan", default="out/plan.json")
    ap.add_argument("--melody", default="out/melody.mid")
    ap.add_argument("--out", default="out")
    ap.add_argument("--accomp-gain", type=float, default=0.6, help="עוצמת הליווי יחסית לקול (RMS)")
    ap.add_argument("--soundfont", default="")
    args = ap.parse_args()

    out = Path(args.out)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    log(f"סגנון: {plan.get('style', {}).get('name')} | קצב: {plan.get('rhythm', {}).get('feel')}")

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        voice_wav = d / "voice.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.audio, "-ac", "2", "-ar", str(SR), str(voice_wav)], check=True)
        voice = load_stereo(voice_wav)
        duration = len(voice) / SR

        pm = build_midi(plan, args.melody, duration)
        midi_path = out / "arrangement.mid"
        pm.write(str(midi_path))

        acc_wav = d / "acc.wav"
        synthesize(pm, midi_path, acc_wav, find_soundfont(args.soundfont))
        acc = load_stereo(acc_wav)

        mixed = mix(voice, acc, args.accomp_gain)
        sf.write(str(d / "mix.wav"), mixed, SR)
        to_mp3(d / "mix.wav", out / "result.mp3")
        to_mp3(acc_wav, out / "accompaniment.mp3")
    log("סיום: result.mp3, accompaniment.mp3, arrangement.mid")


if __name__ == "__main__":
    main()
