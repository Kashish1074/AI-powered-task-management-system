

import pandas as pd
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

IN_PATH = "../outputs/cleaned_tasks.csv"
OUT_PATH = "../outputs/preprocessed_tasks.csv"
COL_DESCRIPTION = "description"

stemmer = PorterStemmer()
STOPWORDS = set(ENGLISH_STOP_WORDS)

TOKEN_RE = re.compile(r"[a-zA-Z]+")


def clean_text(text: str) -> str:
    """Lowercase, tokenize, remove stopwords/punctuation/numbers, stem."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    tokens = TOKEN_RE.findall(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


def main():
    df = pd.read_csv(IN_PATH)
    print(f"Loaded {len(df)} rows")

    df["clean_text"] = df[COL_DESCRIPTION].apply(clean_text)

    # drop rows that became empty after cleaning (pure noise / non-English junk)
    before = len(df)
    df = df[df["clean_text"].str.strip() != ""]
    print(f"Dropped {before - len(df)} rows with empty text after cleaning")

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved preprocessed data -> {OUT_PATH}")

    print("\nSample before/after:")
    for i in range(min(5, len(df))):
        print(f"  RAW : {df[COL_DESCRIPTION].iloc[i]}")
        print(f"  CLEAN: {df['clean_text'].iloc[i]}\n")


if __name__ == "__main__":
    main()