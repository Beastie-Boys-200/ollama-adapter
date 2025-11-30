from typing import List
import os
import glob
from controllers.web_parsing.clean import semantic_clean_texts


def semantic_clean(
    texts: List[str],
    keep_ratio: float = 0.7,
    similarity_threshold: float = 0.9,
    with_log: bool = False,
) -> List[str]:
    """
    Performs semantic cleaning and deduplication of raw text documents.

    Steps:
      1. Split texts into individual sentences.
      2. Compute TF-IDF scores and keep only the most informative sentences.
      3. Cluster semantically similar sentences and remove duplicates.
      4. Rebuild cleaned documents in their original order.

    Args:
        texts (List[str]): List of raw text documents.
        keep_ratio (float): Fraction of top informative sentences to keep (0.0–1.0).
        similarity_threshold (float): Cosine similarity threshold for merging similar sentences (0.85–0.95 typical).
        with_log (bool): If True, prints detailed processing logs.

    Returns:
        List[str]: Cleaned and deduplicated versions of the input texts.
    """
    result = semantic_clean_texts(texts, keep_ratio, similarity_threshold, with_log)
    return result


if __name__ == "__main__":
    input_dir = "logs"
    output_dir = "logs_semantic"

    # === 1. Collect all .txt files ===
    files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
    if not files:
        print(f"No .txt files found in {input_dir}")
        exit()

    # === 2. Read raw documents ===
    raw_texts = []
    for filename in files:
        with open(filename, "r", encoding="utf-8") as f:
            raw_texts.append(f.read())

    print(f"Loaded {len(raw_texts)} text files from '{input_dir}'")

    # === 3. Perform semantic cleaning ===
    cleaned_texts = semantic_clean(
        raw_texts,
        keep_ratio=0.7,
        with_log=True,
    )

    # === 4. Save cleaned documents ===
    print()
    os.makedirs(output_dir, exist_ok=True)

    for src, cleaned_text in zip(files, cleaned_texts):
        base_name = os.path.basename(src)
        out_path = os.path.join(output_dir, base_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text.strip())
        print(f"Saved cleaned file: {base_name} -> {out_path}")

    print(f"\nCleaning complete. Results saved in: '{output_dir}'")
