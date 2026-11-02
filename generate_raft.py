import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "ICLE_rubric_rated_500.csv"
TEXT_DIR = BASE_DIR / "texts"
OUTPUT_PATH = BASE_DIR / "muet_raft_train_argon.jsonl"

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-4-argon")

# Start with 3 to test. Set to None to process all eligible rows.
MAX_ESSAYS = None

# Pause between requests to reduce the chance of hitting rate limits.
REQUEST_DELAY_SECONDS = 2

# Retry failed API calls this many times.
MAX_RETRIES = 3

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY was not found. Add it to the .env file."
    )

client = genai.Client(api_key=API_KEY)


# ============================================================
# 2. PROVISIONAL CEFR-TO-MUET MAPPING
# ============================================================
# IMPORTANT:
# This is a research approximation for initial experiments.
# It is NOT an official CEFR-to-MUET conversion table.
#
# Verify your labels and justify any mapping in your FYP.
# Unknown labels are excluded instead of guessed.

CEFR_TO_MUET = {
    "A1": "Band 1.0",
    "A2": "Band 2.0",
    "B1": "Band 3.0",
    "B1+": "Band 3.0",
    "B2": "Band 4.0",
    "B2+": "Band 4.0",
    "C1": "Band 5.0",
    "C2": "Band 5.0+",
}


# ============================================================
# 3. RUBRIC CONTEXT
# ============================================================

RUBRIC_CONTEXT = """
This is an experimental rubric guide for providing formative feedback
on an English essay.

Evaluate the writing using these broad dimensions:

1. Task fulfilment:
   Does the essay address the prompt and develop relevant ideas?

2. Organisation and coherence:
   Are ideas logically ordered, connected, and supported by paragraphs?

3. Vocabulary:
   Is the word choice appropriate, accurate, and varied?

4. Grammar and sentence structure:
   Are sentence structures understandable and grammatically accurate?

5. Development and support:
   Are claims explained with reasons, examples, or relevant evidence?

The provided CEFR-derived band is a provisional research label.
It is not a verified official MUET score. Do not claim that this
experimental mapping is an official MUET conversion.

Give feedback that helps the learner improve. Do not rewrite the
entire essay unless a short example is needed to explain a correction.
"""


# ============================================================
# 4. FILE AND LABEL HELPERS
# ============================================================

def normalize_label(value):
    """Normalize a CEFR label read from the CSV."""
    if pd.isna(value):
        return None

    label = str(value).strip().upper()

    # Handle values like "B2 " or "CEFR B2".
    label = label.replace("CEFR", "").strip()

    # Only accept labels explicitly present in the mapping.
    if label in CEFR_TO_MUET:
        return label

    return None


def resolve_essay_path(filename):
    """
    Find the essay file using the CSV filename.

    Tries:
    - exact filename
    - filename with .txt appended
    - filename with common extensions removed/appended
    - case-insensitive match
    """
    if pd.isna(filename):
        return None

    raw_name = str(filename).strip()

    if not raw_name:
        return None

    # Prevent a CSV value from pointing outside the texts directory.
    safe_name = Path(raw_name).name

    candidates = [
        TEXT_DIR / safe_name,
        TEXT_DIR / f"{safe_name}.txt",
    ]

    stem = Path(safe_name).stem

    candidates.extend([
        TEXT_DIR / f"{stem}.txt",
        TEXT_DIR / f"{stem}.TXT",
        TEXT_DIR / f"{stem}.essay",
    ])

    for path in candidates:
        if path.is_file():
            return path

    # Last resort: case-insensitive filename match.
    if TEXT_DIR.is_dir():
        wanted_names = {
            safe_name.lower(),
            f"{safe_name}.txt".lower(),
            f"{stem}.txt".lower(),
        }

        for path in TEXT_DIR.iterdir():
            if path.is_file() and path.name.lower() in wanted_names:
                return path

    return None


def load_essay_text(filename):
    """Read the essay text, or return None if no file is found."""
    path = resolve_essay_path(filename)

    if path is None:
        return None

    try:
        text = path.read_text(encoding="utf-8-sig").strip()
    except UnicodeDecodeError:
        # Some text files may have been saved using a legacy encoding.
        text = path.read_text(encoding="cp1252").strip()

    return text if text else None


def load_completed_ids():
    """Read existing output so a rerun does not duplicate successful rows."""
    completed = set()

    if not OUTPUT_PATH.exists():
        return completed

    with OUTPUT_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                essay_id = record.get("essay_id")

                if essay_id is not None:
                    completed.add(str(essay_id))
            except json.JSONDecodeError:
                print("Warning: ignoring a malformed JSONL line.")

    return completed


# ============================================================
# 5. GEMINI FEEDBACK GENERATION
# ============================================================

def build_feedback_prompt(essay, cefr_label, provisional_band):
    return f"""
You are an experienced English writing tutor helping a university
student develop academic essay writing.

Your task is to produce accurate, specific, constructive feedback
on the essay below.

{RUBRIC_CONTEXT}

Provisional source label: {cefr_label}
Experimental mapped band: {provisional_band}

Important rules:
- Treat the band as a supplied research label, not a score you verified.
- Do not claim that the mapping is an official MUET conversion.
- Analyse the actual essay, not an imagined essay.
- Do not invent mistakes or quote wording that is not in the essay.
- When explaining an error, quote a short exact phrase from the essay.
- If a phrase is acceptable, do not incorrectly label it as an error.
- Distinguish clear errors from optional stylistic improvements.
- Be supportive, specific, and suitable for an English learner.
- Do not infer the student's identity, background, or personal traits.
- Do not claim to have applied an official MUET scoring procedure.
- Do not award a new official score.

Return feedback using the following headings:

1. Overall Summary
   Summarise the essay's main strengths and most important areas to improve.

2. Error Analysis
   For each useful example, include:
   - Original phrase (quote exactly)
   - Suggested correction
   - Explanation
   - Whether this is a clear error or an optional improvement

3. Main Writing Weaknesses
   Discuss grammar, vocabulary, sentence structure, coherence,
   task fulfilment, and idea development where relevant.
   Do not force comments on areas that are not problematic.

4. Personalised Improvement Plan
   Give three to five concrete actions the student can practise.

5. Writing Strengths
   Identify genuine strengths supported by examples from the essay.

6. Example Revision
   Revise one short section if it helps demonstrate improvement.
   Explain what changed. Do not rewrite the whole essay.

7. Final Encouragement
   Give a brief, realistic, constructive closing comment.

Essay:
<essay>
{essay}
</essay>
"""


def generate_feedback_gemini(essay, cefr_label, provisional_band):
    """Generate feedback with Gemini, retrying temporary failures."""
    prompt = build_feedback_prompt(
        essay=essay,
        cefr_label=cefr_label,
        provisional_band=provisional_band,
    )

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )

            feedback = (response.text or "").strip()

            if not feedback:
                raise ValueError("Gemini returned an empty response.")

            return feedback

        except Exception as error:
            last_error = error
            wait_seconds = 2 ** attempt

            print(
                f"Gemini request failed "
                f"({attempt}/{MAX_RETRIES}): {error}"
            )

            if attempt < MAX_RETRIES:
                print(f"Retrying in {wait_seconds} seconds...")
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"Gemini failed after {MAX_RETRIES} attempts: {last_error}"
    )


# ============================================================
# 6. CREATE A TRAINING EXAMPLE
# ============================================================

def make_training_record(
    essay_id,
    essay,
    cefr_label,
    provisional_band,
    feedback,
):
    """
    Use a chat-message format so the data can later be formatted
    using the selected Llama model's chat template.
    """
    user_prompt = f"""
{RUBRIC_CONTEXT}

Analyse the following English essay and provide detailed, personalised
formative feedback.

Source CEFR label: {cefr_label}
Experimental mapped band: {provisional_band}

Essay:
{essay}
""".strip()

    assistant_response = f"""
Provisional research label: {provisional_band}

{feedback}
""".strip()

    return {
        "essay_id": str(essay_id),
        "source": "ICLE_with_Gemini_generated_feedback",
        "label_type": "provisional_CEFR_to_MUET_mapping",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an English essay feedback assistant. "
                    "Give accurate, specific, constructive feedback. "
                    "Do not invent errors. Clearly distinguish "
                    "language errors from optional style improvements. "
                    "Never claim an unverified score is official."
                ),
            },
            {
                "role": "user",
                "content": user_prompt,
            },
            {
                "role": "assistant",
                "content": assistant_response,
            },
        ],
    }


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def main():
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    if not TEXT_DIR.is_dir():
        raise FileNotFoundError(f"Essay folder not found: {TEXT_DIR}")

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

    required_columns = {"filename", "CEFR_MFRM"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing_columns)}\n"
            f"Actual columns: {list(df.columns)}"
        )

    print(f"CSV rows: {len(df)}")
    print(f"Text directory: {TEXT_DIR}")
    print(f"Gemini model: {MODEL_NAME}")

    completed_ids = load_completed_ids()
    print(f"Already generated: {len(completed_ids)}")

    generated_count = 0
    skipped_count = 0
    failed_count = 0
    attempted_count = 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for row_index, row in df.iterrows():
        filename = row["filename"]
        essay_id = str(filename).strip()

        if not essay_id or essay_id.lower() == "nan":
            print(f"[SKIP] Row {row_index}: missing filename")
            skipped_count += 1
            continue

        if essay_id in completed_ids:
            continue

        cefr_label = normalize_label(row["CEFR_MFRM"])

        if cefr_label is None:
            print(
                f"[SKIP] {essay_id}: unknown CEFR_MFRM "
                f"value {row['CEFR_MFRM']!r}"
            )
            skipped_count += 1
            continue

        essay = load_essay_text(filename)

        if not essay:
            print(f"[SKIP] Essay file not found or empty: {filename}")
            skipped_count += 1
            continue

        if len(essay) < 50:
            print(f"[SKIP] Essay appears too short: {filename}")
            skipped_count += 1
            continue

        provisional_band = CEFR_TO_MUET[cefr_label]

        # Stop after MAX_ESSAYS newly attempted essays when testing.
        if MAX_ESSAYS is not None and attempted_count >= MAX_ESSAYS:
            break

        attempted_count += 1
        print(
            f"\n[{attempted_count}] Generating feedback for "
            f"{filename} ({cefr_label} -> {provisional_band})"
        )

        try:
            feedback = generate_feedback_gemini(
                essay=essay,
                cefr_label=cefr_label,
                provisional_band=provisional_band,
            )

            record = make_training_record(
                essay_id=essay_id,
                essay=essay,
                cefr_label=cefr_label,
                provisional_band=provisional_band,
                feedback=feedback,
            )

            # Append each completed record immediately.
            with OUTPUT_PATH.open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

            completed_ids.add(essay_id)
            generated_count += 1

            print(f"[SAVED] {OUTPUT_PATH.name}")

        except Exception as error:
            failed_count += 1
            print(f"[FAILED] {filename}: {error}")

        time.sleep(REQUEST_DELAY_SECONDS)

    print("\n========== SUMMARY ==========")
    print(f"New records generated: {generated_count}")
    print(f"Rows skipped:          {skipped_count}")
    print(f"Rows failed:           {failed_count}")
    print(f"Output file:           {OUTPUT_PATH}")
    print("=============================")


if __name__ == "__main__":
    main()