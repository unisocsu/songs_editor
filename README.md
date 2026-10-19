# song-edit

עריכת שירים על CPU בלבד (GitHub Actions), עם שמירה על הלחן המקורי.

## רעיון
הלחן נחלץ כנתונים (MIDI) ונעול. רק מה שהתבקש משתנה.

| שלב | מצב | פלט |
|-----|------|------|
| 1. ניתוח | ✅ קיים | `analysis.json`, `melody.mid`, `stems/`, `summary.md` |
| 2. תכנון (Gemini: הנחיה -> JSON פעולות) | ⏳ הבא | `plan.json` |
| 3. יצירה / עיבוד מחדש | ⏳ הבא | `result.mp3` |
| 4. בדיקת נאמנות ללחן | ⏳ הבא | דוח דמיון |

## שימוש
1. העלה שיר לתיקיית `songs/` (או השתמש בכל קישור ישיר).
2. Actions -> **analyze-song** -> Run workflow.
3. הדבק את הקישור לקובץ. לדוגמה:
   `https://github.com/USER/REPO/blob/main/songs/shir.mp3` (מומר אוטומטית ל-raw)
   או נתיב בתוך הריפו: `songs/shir.mp3`
4. בסיום: Artifacts -> `analysis` (zip עם כל הפלט).

> ריפו פרטי: קישור URL לא יעבוד בלי טוקן. השתמש בנתיב בתוך הריפו.

## הרצה מקומית
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python src/analyze.py songs/shir.mp3 --out out            # מלא
python src/analyze.py songs/shir.mp3 --out out --no-stems # מהיר
python src/analyze.py songs/shir.mp3 --out out --lyrics   # + תמלול
```

## דגלים נוספים
- `--tempo 92` לכפות BPM אם הזיהוי האוטומטי יצא כפול/חצי (ב-`analysis.json` יש `tempo_alternatives`)
- `--lyrics-lang he` שפת התמלול (ברירת מחדל עברית; בלי זה Whisper עלול לזהות אנגלית ולתעתק)

## הערות
- הניתוח אלגוריתמי (הערכה). מומלץ לבדוק את `melody.mid` לפני שלבי היצירה.
- `pyin` מתאים למנגינה חד-קולית (שירה). `basic-pitch` לחומר פוליפוני.
- ערוץ השירה נשאר ערוץ אודיו מקורי; MIDI מתאר תווים, לא מילים או גוון קול.
- ריצה מלאה עם Demucs: כמה דקות עד כמה עשרות דקות לשיר, תלוי באורך.
