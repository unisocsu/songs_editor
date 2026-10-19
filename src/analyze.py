#!/usr/bin/env python3
"""שלב 1: ניתוח שיר.

קלט:  קובץ אודיו כלשהו (mp3 / wav / m4a ...)
פלט (בתיקיית --out):
    analysis.json   קצב, סולם, אקורדים, מבנה, אנרגיה, סיכום מנגינה (+ תמלול אם התבקש)
    melody.mid      המנגינה כקובץ MIDI (הלחן "הננעל")
    stems/*.wav     vocals / drums / bass / other  (אם לא ביקשת --no-stems)
    summary.md      סיכום קריא
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf

SR = 22050
HOP = 512
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# פרופילי Krumhansl-Schmuckler לזיהוי סולם
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def log(msg):
    print(f"[analyze] {msg}", flush=True)


# ---------------------------------------------------------------- קלט / stems
def to_wav(src: Path, dst: Path):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "2", "-ar", "44100", str(dst)],
        check=True,
    )


def separate_stems(wav: Path, out_dir: Path) -> dict:
    """Demucs על CPU. מחזיר {שם: נתיב}."""
    work = out_dir / "_demucs"
    cmd = [sys.executable, "-m", "demucs", "-n", "htdemucs", "-d", "cpu", "-o", str(work), str(wav)]
    log("מפריד ערוצים עם Demucs (CPU) - זה לוקח כמה דקות...")
    subprocess.run(cmd, check=True)
    src_dir = work / "htdemucs" / wav.stem
    stems_dir = out_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for name in ("vocals", "drums", "bass", "other"):
        f = src_dir / f"{name}.wav"
        if f.exists():
            shutil.copy(f, stems_dir / f"{name}.wav")
            result[name] = stems_dir / f"{name}.wav"
    shutil.rmtree(work, ignore_errors=True)
    return result


# ---------------------------------------------------------------------- ניתוח
def detect_tempo_beats(y, sr):
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP)
    tempo = float(np.atleast_1d(tempo)[0])
    return tempo, np.asarray(beat_frames, dtype=int)


def detect_key(y_harm, sr):
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=HOP).mean(axis=1)
    best = None
    for i in range(12):
        for mode, prof in (("major", KS_MAJOR), ("minor", KS_MINOR)):
            c = float(np.corrcoef(chroma, np.roll(prof, i))[0, 1])
            if best is None or c > best[0]:
                best = (c, i, mode)
    return {"tonic": NOTE_NAMES[best[1]], "mode": best[2], "confidence": round(best[0], 3)}


def chord_templates():
    names, templates = [], []
    for i in range(12):
        for suffix, intervals in (("", (0, 4, 7)), ("m", (0, 3, 7))):
            v = np.zeros(12)
            for x in intervals:
                v[(i + x) % 12] = 1.0
            names.append(NOTE_NAMES[i] + suffix)
            templates.append(v / np.linalg.norm(v))
    return names, np.array(templates)


def beat_bounds(beat_frames, n_frames):
    return librosa.util.fix_frames(beat_frames, x_min=0, x_max=n_frames - 1)


def sync_mean(feat, bounds):
    cols = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        b = max(b, a + 1)
        cols.append(feat[:, a:b].mean(axis=1))
    return np.array(cols).T


def detect_chords(chroma_sync, times):
    names, templates = chord_templates()
    seq = []
    for i in range(chroma_sync.shape[1]):
        col = chroma_sync[:, i]
        n = np.linalg.norm(col)
        if n < 1e-6:
            seq.append("N")
            continue
        seq.append(names[int(np.argmax(templates @ (col / n)))])
    merged = []
    for i, ch in enumerate(seq):
        start, end = float(times[i]), float(times[i + 1])
        if merged and merged[-1]["chord"] == ch:
            merged[-1]["end"] = round(end, 3)
        else:
            merged.append({"start": round(start, 3), "end": round(end, 3), "chord": ch})
    return merged


def detect_structure(chroma_sync, mfcc_sync, times, duration):
    n = chroma_sync.shape[1]
    if n < 16:
        return [{"start": 0.0, "end": round(duration, 3), "label": "A"}]
    feats = np.vstack([chroma_sync, mfcc_sync])
    feats = (feats - feats.mean(axis=1, keepdims=True)) / (feats.std(axis=1, keepdims=True) + 1e-8)
    k = int(np.clip(round(duration / 20), 3, 10))
    k = min(k, n // 4)
    bounds = librosa.segment.agglomerative(feats, k)
    bounds = sorted(set(int(b) for b in bounds) | {0})
    edges = bounds + [n]
    means = [feats[:, a:b].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])]
    labels, label_vecs = [], []
    for m in means:
        found = None
        for li, v in enumerate(label_vecs):
            cos = float(m @ v / (np.linalg.norm(m) * np.linalg.norm(v) + 1e-8))
            if cos > 0.6:
                found = li
                break
        if found is None:
            label_vecs.append(m)
            found = len(label_vecs) - 1
        labels.append(chr(ord("A") + found))
    out = []
    for (a, b), lab in zip(zip(edges[:-1], edges[1:]), labels):
        end_t = duration if b >= n else float(times[b])
        out.append({"start": round(float(times[a]), 3), "end": round(end_t, 3), "label": lab})
    return out


def energy_curve(y, sr):
    hop = sr  # שנייה אחת
    rms = librosa.feature.rms(y=y, frame_length=hop, hop_length=hop)[0]
    rms = rms / (rms.max() + 1e-9)
    return [round(float(x), 3) for x in rms]


# ---------------------------------------------------------------------- מנגינה
def melody_to_midi_pyin(y, sr, bpm, out_path: Path):
    """מנגינה חד-קולית (שירה) -> MIDI באמצעות pyin."""
    log("מחלץ מנגינה (pyin)...")
    f0, voiced, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"),
        sr=sr, hop_length=HOP, frame_length=2048,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=HOP)
    frame_dur = HOP / sr
    midi_vals = np.where(voiced & ~np.isnan(f0), np.round(librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0))), np.nan)

    notes, cur = [], None  # cur = [pitch, start, end]
    for t, m in zip(times, midi_vals):
        if np.isnan(m):
            if cur:
                notes.append(cur)
                cur = None
            continue
        if cur and cur[0] == m:
            cur[2] = t + frame_dur
        else:
            if cur:
                notes.append(cur)
            cur = [m, t, t + frame_dur]
    if cur:
        notes.append(cur)
    notes = [n for n in notes if n[2] - n[1] >= 0.08]

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=0, name="melody")
    for p, s, e in notes:
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=int(p), start=float(s), end=float(e)))
    pm.instruments.append(inst)
    pm.write(str(out_path))
    return notes


def melody_to_midi_basic_pitch(path: Path, out_path: Path):
    from basic_pitch.inference import predict

    log("מחלץ MIDI (basic-pitch)...")
    _, midi, note_events = predict(str(path))
    midi.write(str(out_path))
    return [[n[2], n[0], n[1]] for n in note_events]  # pitch, start, end


def melody_summary(notes):
    if not notes:
        return {"note_count": 0}
    pitches = [int(n[0]) for n in notes]
    return {
        "note_count": len(notes),
        "lowest": pretty_midi.note_number_to_name(min(pitches)),
        "highest": pretty_midi.note_number_to_name(max(pitches)),
    }


# ---------------------------------------------------------------------- תמלול
def transcribe(path: Path):
    from faster_whisper import WhisperModel

    log("מתמלל מילים (faster-whisper, CPU)...")
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(path), word_timestamps=True, vad_filter=True)
    out = []
    for s in segments:
        out.append({
            "start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
            "words": [{"w": w.word.strip(), "start": round(w.start, 2), "end": round(w.end, 2)} for w in (s.words or [])],
        })
    return {"language": info.language, "segments": out}


# ------------------------------------------------------------------------ main
def fmt_t(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


def write_summary(a, path: Path):
    lines = [
        "# ניתוח שיר",
        "",
        f"- משך: {fmt_t(a['duration_sec'])}",
        f"- קצב: {a['tempo_bpm']:.1f} BPM",
        f"- סולם: {a['key']['tonic']} {a['key']['mode']} (ביטחון {a['key']['confidence']})",
        f"- מנגינה: {a['melody']}",
        "",
        "## מבנה",
    ]
    for s in a["structure"]:
        lines.append(f"- {fmt_t(s['start'])}-{fmt_t(s['end'])}: חלק {s['label']}")
    lines += ["", "## אקורדים (20 ראשונים)"]
    lines.append(", ".join(f"{c['chord']}@{fmt_t(c['start'])}" for c in a["chords"][:20]))
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", default="out")
    ap.add_argument("--no-stems", action="store_true", help="דלג על הפרדת ערוצים (מהיר)")
    ap.add_argument("--midi-engine", choices=["pyin", "basic-pitch"], default="pyin")
    ap.add_argument("--lyrics", action="store_true", help="תמלול מילים עם זמנים")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.input)

    wav = out / "input.wav"
    log("ממיר ל-WAV...")
    to_wav(src, wav)

    stems = {}
    if not args.no_stems:
        stems = separate_stems(wav, out)

    log("טוען אודיו לניתוח...")
    y, sr = librosa.load(str(wav), sr=SR, mono=True)
    duration = float(len(y) / sr)

    log("קצב וביטים...")
    tempo, beat_frames = detect_tempo_beats(y, sr)
    y_harm = librosa.effects.harmonic(y)

    log("סולם, אקורדים ומבנה...")
    key = detect_key(y_harm, sr)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP, n_mfcc=13)
    bounds = beat_bounds(beat_frames, chroma.shape[1])
    times = librosa.frames_to_time(bounds, sr=sr, hop_length=HOP)
    chroma_sync = sync_mean(chroma, bounds)
    mfcc_sync = sync_mean(mfcc, bounds)
    chords = detect_chords(chroma_sync, times)
    structure = detect_structure(chroma_sync, mfcc_sync, times, duration)
    energy = energy_curve(y, sr)

    # מנגינה: מערוץ השירה אם קיים, אחרת מהמיקס
    melody_path = out / "melody.mid"
    if args.midi_engine == "basic-pitch":
        notes = melody_to_midi_basic_pitch(stems.get("vocals", wav), melody_path)
    else:
        if "vocals" in stems:
            yv, _ = librosa.load(str(stems["vocals"]), sr=SR, mono=True)
        else:
            yv = y
        notes = melody_to_midi_pyin(yv, SR, tempo, melody_path)

    analysis = {
        "source_file": src.name,
        "duration_sec": round(duration, 3),
        "tempo_bpm": round(tempo, 2),
        "beats_sec": [round(float(t), 3) for t in librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)],
        "key": key,
        "chords": chords,
        "structure": structure,
        "energy_per_sec": energy,
        "melody": melody_summary(notes),
        "files": {
            "melody_midi": "melody.mid",
            "stems": {k: f"stems/{k}.wav" for k in stems},
        },
        "notes": "ניתוח אלגוריתמי - הערכה בלבד. מומלץ לבדוק את melody.mid לפני שלבי היצירה.",
    }
    if args.lyrics:
        analysis["lyrics"] = transcribe(stems.get("vocals", wav))

    (out / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(analysis, out / "summary.md")
    wav.unlink(missing_ok=True)
    log(f"סיום. הפלט נמצא ב-{out}/")


if __name__ == "__main__":
    main()
