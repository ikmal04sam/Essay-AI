
import pandas as pd
import ollama
import os
import random
import json

# ============================================================
# 1. LOAD DATASET AND MATCH ESSAY TEXTS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "ICLE_rubric_rated_500.csv")
TEXT_DIR = os.path.join(BASE_DIR, "texts")
OUTPUT_PATH = os.path.join(BASE_DIR, "muet_raft_train.jsonl")

# Load CSV
metadata_df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

print("CSV loaded successfully.")
print("Total CSV rows:", len(metadata_df))
print("Available columns:", metadata_df.columns.tolist())

# Check required columns
required_columns = ["filename", "CEFR_MFRM"]

missing_columns = [
    col for col in required_columns
    if col not in metadata_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required CSV columns: {missing_columns}"
    )


def load_essay_text(filename):
    """Load an essay using its filename from the CSV."""

    if pd.isna(filename):
        return None

    filename = str(filename).strip()

    # Try the exact filename first
    candidates = [filename]

    # If the filename has no extension, also try .txt
    if not os.path.splitext(filename)[1]:
        candidates.append(filename + ".txt")

    for candidate in candidates:
        file_path = os.path.join(TEXT_DIR, candidate)

        if os.path.isfile(file_path):
            try:
                with open(
                    file_path, "r", encoding="utf-8-sig"
                ) as f:
                    essay = f.read().strip()

                if essay:
                    return essay

            except (OSError, UnicodeError) as e:
                print(f"Error reading {file_path}: {e}")
                return None

    return None


# Match each CSV row to its essay text
metadata_df["essay_text"] = metadata_df["filename"].apply(
    load_essay_text
)

matched_count = metadata_df["essay_text"].notna().sum()
missing_count = metadata_df["essay_text"].isna().sum()

print("\nEssay matching results")
print("----------------------")
print("CSV rows:", len(metadata_df))
print("Essays matched:", matched_count)
print("Essays missing:", missing_count)

if missing_count > 0:
    print("\nExamples of unmatched filenames:")
    print(
        metadata_df.loc[
            metadata_df["essay_text"].isna(), "filename"
        ].head(10).to_list()
    )

# Keep rows with essay text and a CEFR score
dataset_df = metadata_df.dropna(
    subset=["essay_text", "CEFR_MFRM"]
).copy()

if dataset_df.empty:
    raise ValueError(
        "No essays were matched. Check the texts folder "
        "and the filenames in your CSV."
    )

print("Essays ready for processing:", len(dataset_df))


# ============================================================
# 2. MAP CEFR LEVELS TO MUET BANDS
# ============================================================

def map_cefr_to_muet(cefr_score):
    mapping = {
        "A1": "Band 1.0",
        "A2": "Band 2.0",
        "B1": "Band 3.0",
        "B2": "Band 4.0",
        "C1": "Band 5.0",
        "C2": "Band 5.0+"
    }

    if pd.isna(cefr_score):
        return None

    cefr_score = str(cefr_score).strip().upper()
    return mapping.get(cefr_score)


dataset_df["MUET_Band"] = dataset_df["CEFR_MFRM"].apply(
    map_cefr_to_muet
)

# Do not silently assign Band 3.0 to unknown CEFR labels
invalid_scores = dataset_df["MUET_Band"].isna()

if invalid_scores.any():
    print("\nWarning: Unrecognized CEFR labels:")
    print(
        dataset_df.loc[
            invalid_scores, "CEFR_MFRM"
        ].value_counts(dropna=False)
    )

dataset_df = dataset_df.dropna(subset=["MUET_Band"]).copy()

if dataset_df.empty:
    raise ValueError(
        "No valid CEFR scores could be mapped to MUET bands. "
        "Check the values in the CEFR_MFRM column."
    )

print("Essays with valid MUET mappings:", len(dataset_df))
print("\nMUET band distribution:")
print(dataset_df["MUET_Band"].value_counts())


# ============================================================
# 3. GENERATE SYNTHETIC FEEDBACK USING OLLAMA
# ============================================================

def generate_feedback_ollama(essay, muet_band):
    prompt = f"""
You are an expert Malaysian University English Test (MUET)
writing assessor.

The dataset assigns this student essay the following provisional
MUET band: {muet_band}.

Write concise feedback in 3-4 sentences addressed directly
to the student.

Discuss:
1. Task fulfilment and development of ideas.
2. Grammar and language accuracy.
3. Vocabulary and sentence organization.
4. One strength and one specific area for improvement.

Do not rewrite the essay.

Treat the assigned band as a dataset label, not proof that
the band is correct. Base your feedback on the essay and avoid
inventing errors or strengths.

Student essay:
{essay}
"""

    try:
        response = ollama.chat(
            model="llama3.1:8b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        feedback = response["message"]["content"].strip()

        if not feedback:
            print("Warning: Ollama returned empty feedback.")
            return None

        return feedback

    except Exception as e:
        print(f"Ollama error: {e}")
        return None


print("\nGenerating MUET feedback...")
print("This may take a while.")

feedback_results = []

for index, row in dataset_df.iterrows():
    print(
        f"Processing essay {len(feedback_results) + 1}"
        f"/{len(dataset_df)}: {row['filename']}"
    )

    feedback = generate_feedback_ollama(
        row["essay_text"],
        row["MUET_Band"]
    )

    feedback_results.append(feedback)

dataset_df["muet_feedback"] = feedback_results

successful_count = dataset_df["muet_feedback"].notna().sum()
failed_count = dataset_df["muet_feedback"].isna().sum()

print("\nFeedback generation results")
print("----------------------------")
print("Successful:", successful_count)
print("Failed:", failed_count)

# Exclude rows without generated feedback
dataset_df = dataset_df.dropna(
    subset=["muet_feedback"]
).copy()

if dataset_df.empty:
    raise ValueError(
        "No feedback was generated successfully. "
        "Check that Ollama is running and llama3.1 is installed."
    )


# ============================================================
# 4. DEFINE MUET RUBRIC KNOWLEDGE
# ============================================================

muet_rubric = {
    "Band 1.0": (
        "MUET Band 1: Highly limited language proficiency. "
        "Very limited control of language; frequent severe "
        "errors may hinder meaning."
    ),
    "Band 2.0": (
        "MUET Band 2: Limited language proficiency. "
        "Limited control; grammar and vocabulary errors "
        "may frequently obscure meaning."
    ),
    "Band 3.0": (
        "MUET Band 3: Satisfactory language proficiency. "
        "Some noticeable errors, but meaning is generally clear."
    ),
    "Band 4.0": (
        "MUET Band 4: Competent language proficiency. "
        "Generally good language control and vocabulary, "
        "with occasional errors."
    ),
    "Band 5.0": (
        "MUET Band 5: Very good language proficiency. "
        "High grammatical accuracy, effective vocabulary, "
        "and well-organized ideas."
    ),
    "Band 5.0+": (
        "MUET Band 5+: Highly proficient language use. "
        "Strong control of complex language, precise vocabulary, "
        "and effective organization."
    )
}


# ============================================================
# 5. FORMAT EXAMPLES FOR RAFT
# ============================================================

def format_for_raft(row):
    essay = row["essay_text"]
    score = row["MUET_Band"]
    feedback = row["muet_feedback"]

    # 80% correct rubric context, 20% distractor context
    if random.random() < 0.8:
        context = muet_rubric[score]
    else:
        distractor_scores = [
            band for band in muet_rubric
            if band != score
        ]

        wrong_band = random.choice(distractor_scores)
        context = muet_rubric[wrong_band]

    raft_prompt = f"""
Assess the following student essay.

Use the supplied rubric context as retrieved reference
material. Evaluate the essay carefully and do not assume
that the context is necessarily correct for the essay.

<context>
{context}
</context>

Student Essay:
{essay}
"""

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert MUET essay feedback assistant. "
                    "Provide constructive, evidence-based feedback. "
                    "Use the essay and rubric context carefully. "
                    "Do not invent evidence about the essay."
                )
            },
            {
                "role": "user",
                "content": raft_prompt.strip()
            },
            {
                "role": "assistant",
                "content": (
                    f"**MUET Band:** {score}\n\n"
                    f"**Feedback:** {feedback}"
                )
            }
        ]
    }


raft_dataset = dataset_df.apply(
    format_for_raft, axis=1
).tolist()


# ============================================================
# 6. SAVE JSONL DATASET
# ============================================================

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in raft_dataset:
        f.write(
            json.dumps(item, ensure_ascii=False) + "\n"
        )

print("\nRAFT dataset saved successfully.")
print("Output file:", OUTPUT_PATH)
print("Total JSONL examples:", len(raft_dataset))
