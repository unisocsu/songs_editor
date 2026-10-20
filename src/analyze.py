#!/usr/bin/env python3
"""שלב 1: חילוץ הלחן מהקלטת סולו (ללא ליווי).

קלט:  קובץ אודיו של שירה / סולו בלי ליווי (mp3 / wav / m4a ...)
פלט (בתיקיית --out):
    melody.mid          הלחן כתווים + Pitch Bend (שומר גלישות וקישוטים) + עוצמה
    pitch_contour.csv   עקום גובה הצליל המלא (Hz, MIDI עשרוני, עוצמה) - המקור המדויק
    analysis.json       סיכום: טווח, שימוש בצלילים, משך, תמלול (אופציונלי)
    summary.md          סיכום קריא

אם בהקלטה יש גם ליווי, הוסף --separate והשירה תופרד עם Demucs לפני החילוץ.
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
from scipy.signal import medfilt

SR = 22050
HOP = 512
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def log(msg):
    print(f"[analyze] {msg}", flush=True)


# ---------------------------------------------------------------- קלט / הפרדה
def to_wav(src: Path, dst: Path):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "44100", str(dst)],
        check=True,
    )


def separate_vocals(wav: Path, out_dir: Path) -> Path:
    """Demucs על CPU - רק כשיש ליווי. מחזיר נתיב ערוץ השירה."""
    work = out_dir / "_demucs"
    log("מפריד שירה מהליווי עם Demucs (CPU) - כמה דקות...")
    subprocess.run(
        [sys.executable, "-m", "demucs", "-n", "htdemucs", "--two-stems", "vocals",
         "-d", "cpu", "-o", str(work), str(wav)],
        check=True,
    )
    src = work / "htdemucs" / wav.stem / "vocals.wav"
    dst = out_dir / "vocals.wav"
    shutil.copy(src, dst)
    shutil.rmtree(work, ignore_errors=True)
    return dst


# ------------------------------------------------------------------ חילוץ גובה
def extract_pitch(y, sr):
    log("מחלץ עקום גובה צליל (pyin)...")
    f0, voiced, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"),
        sr=sr, hop_length=HOP, frame_length=2048,
    )
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]
    n = min(len(f0), len(rms))
    f0, voiced, rms = f0[:n], voiced[:n], rms[:n]
    times = librosa.times_like(f0, sr=sr, hop_length=HOP)
    midi = np.where(voiced & ~np.isnan(f0), librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0)), np.nan)
    return times, f0, midi, rms


def smooth_pitch(midi, k=5):
    """פילטר חציוני על כל קטע קולי רציף. שומר על גלישות, מבטל קפיצות בודדות."""
    out = midi.copy()
    isv = ~np.isnan(out)
    i = 0
    while i < len(out):
        if not isv[i]:
            i += 1
            continue
        j = i
        while j < len(out) and isv[j]:
            j += 1
        if j - i >= k:
            out[i:j] = medfilt(out[i:j], k)
        i = j
    return out


def segment_notes(times, midi_s, rms, min_dur=0.1, merge_gap=0.15):
    """עקום גובה -> תווים (עיגול לחצי טון), עם עוצמה לכל תו."""
    frame = HOP / SR
    rounded = np.round(midi_s)
    raw, cur = [], None
    for i, (t, m) in enumerate(zip(times, rounded)):
        if np.isnan(m):
            if cur:
                raw.append(cur)
                cur = None
            continue
        if cur and cur["pitch"] == m:
            cur["end"] = t + frame
            cur["last"] = i
        else:
            if cur:
                raw.append(cur)
            cur = {"pitch": int(m), "start": float(t), "end": float(t + frame), "first": i, "last": i}
    if cur:
        raw.append(cur)

    notes = [n for n in raw if n["end"] - n["start"] >= min_dur]
    merged = []
    for n in notes:
        if merged and merged[-1]["pitch"] == n["pitch"] and n["start"] - merged[-1]["end"] <= merge_gap:
            merged[-1]["end"] = n["end"]
            merged[-1]["last"] = n["last"]
        else:
            merged.append(dict(n))
    merged = [n for n in merged if n["end"] - n["start"] >= min_dur]

    ref = np.percentile(rms[rms > 0], 95) if np.any(rms > 0) else 1.0
    for n in merged:
        level = float(np.mean(rms[n["first"]: n["last"] + 1])) / (ref + 1e-9)
        n["velocity"] = int(np.clip(40 + 80 * level, 30, 127))
    return merged


def write_midi(notes, times, midi_s, out_path: Path, bend_range=2.0, step=2):
    """MIDI עם Pitch Bend: התווים מעוגלים, הגלישות והקישוטים נשמרים בכיפוף."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, name="melody")
    for n in notes:
        inst.notes.append(pretty_midi.Note(velocity=n["velocity"], pitch=n["pitch"],
                                           start=n["start"], end=n["end"]))
        for k, i in enumerate(range(n["first"], n["last"] + 1, step)):
            if np.isnan(midi_s[i]):
                continue
            dev = float(np.clip(midi_s[i] - n["pitch"], -bend_range, bend_range))
            val = int(np.clip(dev / bend_range * 8191, -8192, 8191))
            t = max(0.0, float(times[i]) - (0.002 if k == 0 else 0.0))
            inst.pitch_bends.append(pretty_midi.PitchBend(val, t))
    pm.instruments.append(inst)
    pm.write(str(out_path))


def write_contour(times, f0, midi, rms, out_path: Path):
    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_sec", "f0_hz", "midi_float", "rms"])
        for t, f, m, r in zip(times, f0, midi, rms):
            w.writerow([f"{t:.4f}",
                        "" if np.isnan(f) or np.isnan(m) else f"{f:.2f}",
                        "" if np.isnan(m) else f"{m:.3f}",
                        f"{r:.5f}"])


def melody_summary(notes):
    if not notes:
        return {"note_count": 0}
    pitches = [n["pitch"] for n in notes]
    pc = {}
    for n in notes:
        name = NOTE_NAMES[n["pitch"] % 12]
        pc[name] = pc.get(name, 0.0) + (n["end"] - n["start"])
    pc = {k: round(v, 2) for k, v in sorted(pc.items(), key=lambda kv: -kv[1])}
    return {
        "note_count": len(notes),
        "lowest": pretty_midi.note_number_to_name(min(pitches)),
        "highest": pretty_midi.note_number_to_name(max(pitches)),
        "voiced_seconds": round(sum(n["end"] - n["start"] for n in notes), 1),
        "pitch_class_seconds": pc,
    }


# ---------------------------------------------------------------------- תמלול
def transcribe(path: Path, lang=None):
    from faster_whisper import WhisperModel

    log("מתמלל מילים (faster-whisper, CPU)...")
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(path), language=lang, word_timestamps=True, vad_filter=True)
    out = []
    for s in segments:
        out.append({
            "start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
            "words": [{"w": w.word.strip(), "start": round(w.start, 2), "end": round(w.end, 2)}
                      for w in (s.words or [])],
        })
    return {"language": info.language, "segments": out}


# ------------------------------------------------------------------------ main
def fmt_t(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


def write_summary(a, path: Path):
    m = a["melody"]
    lines = [
        "# חילוץ לחן",
        "",
        f"- משך: {fmt_t(a['duration_sec'])}",
        f"- תווים: {m.get('note_count', 0)} | טווח: {m.get('lowest', '-')} עד {m.get('highest', '-')}",
        f"- זמן שירה: {m.get('voiced_seconds', 0)} שניות",
        f"- שימוש בצלילים (שניות): {m.get('pitch_class_seconds', {})}",
        "",
        "קבצים: `melody.mid` (תווים + Pitch Bend), `pitch_contour.csv` (המקור המדויק), `analysis.json`.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default="out")
    ap.add_argument("--separate", action="store_true", help="יש ליווי בהקלטה - הפרד שירה עם Demucs")
    ap.add_argument("--lyrics", action="store_true", help="תמלול מילים עם זמנים")
    ap.add_argument("--lyrics-lang", default="he", help="שפת התמלול (ברירת מחדל: he)")
    ap.add_argument("--bend-range", type=float, default=2.0, help="טווח Pitch Bend בחצאי טון")
    ap.add_argument("--min-note", type=float, default=0.1, help="משך תו מינימלי בשניות")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.input)

    wav = out / "input.wav"
    log("ממיר ל-WAV...")
    to_wav(src, wav)
    voice = separate_vocals(wav, out) if args.separate else wav

    y, sr = librosa.load(str(voice), sr=SR, mono=True)
    duration = float(len(y) / sr)

    times, f0, midi, rms = extract_pitch(y, sr)
    midi_s = smooth_pitch(midi)
    notes = segment_notes(times, midi_s, rms, min_dur=args.min_note)

    write_midi(notes, times, midi_s, out / "melody.mid", bend_range=args.bend_range)
    write_contour(times, f0, midi_s, rms, out / "pitch_contour.csv")

    analysis = {
        "source_file": src.name,
        "duration_sec": round(duration, 3),
        "melody": melody_summary(notes),
        "notes": [{"pitch": n["pitch"], "start": round(n["start"], 3), "end": round(n["end"], 3),
                   "velocity": n["velocity"]} for n in notes],
        "pitch_bend_range_semitones": args.bend_range,
        "files": {"melody_midi": "melody.mid", "pitch_contour": "pitch_contour.csv"},
        "remark": "חילוץ אלגוריתמי - הערכה. רצוי לבדוק את melody.mid לפני שלב היצירה.",
    }
    if args.lyrics:
        analysis["lyrics"] = transcribe(voice, args.lyrics_lang)

    (out / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(analysis, out / "summary.md")
    if voice != wav:
        pass  # ערוץ השירה (vocals.wav) נשאר בפלט
    wav.unlink(missing_ok=True)
    log(f"סיום. הפלט נמצא ב-{out}/")


if __name__ == "__main__":
    main()
