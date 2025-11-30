from typing import List
from .src.utils import split_into_sentences, rebuild_docs
from .src.tfidf import compute_sentence_tfidf
from .src.cluster import cluster_similar_sentences


def semantic_clean_texts(
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
    if not texts:
        return []

    # Step 1: Sentence-level preprocessing
    if with_log:
        print("\n--- Sentence splitting ---")

    sentences, sentence_doc_ids = split_into_sentences(texts, with_log=with_log)

    # Step 2: TF-IDF filtering
    if with_log:
        print("\n--- TF-IDF filtering ---")

    informative_sentences, X, _ = compute_sentence_tfidf(
        sentences, keep_ratio=keep_ratio, with_log=with_log
    )

    # Step 3: Clustering for deduplication
    if with_log:
        print("\n--- Clustering for deduplication ---")

    unique_sentences = cluster_similar_sentences(
        informative_sentences,
        X,
        similarity_threshold=similarity_threshold,
        with_log=with_log,
    )

    # Step 4: Rebuild cleaned documents
    cleaned_docs = rebuild_docs(unique_sentences, sentence_doc_ids, len(texts))

    if with_log:
        print(f"\nCleaning complete — {len(cleaned_docs)} cleaned documents ready.")

    return cleaned_docs
