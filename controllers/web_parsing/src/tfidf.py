from typing import List, Tuple
import numpy as np
from statistics import mean, stdev
from sklearn.feature_extraction.text import TfidfVectorizer


def compute_sentence_tfidf(
    sentences: List[str], keep_ratio: float = 0.7, with_log: bool = False
) -> Tuple[List[Tuple[str, int]], np.ndarray, np.ndarray]:
    """
    Computes TF-IDF scores for sentences and selects the most informative ones
    based on a percentile threshold.

    Args:
        sentences (List[str]): List of sentences to analyze.
        keep_ratio (float): Fraction of sentences to keep (0.0–1.0),
            e.g., 0.7 = keep 70% of sentences with the highest TF-IDF scores.
        with_log (bool): If True, prints processing statistics and thresholds.

    Returns:
        Tuple[List[Tuple[str, int]], np.ndarray, np.ndarray]:
            - informative: List of (sentence, index) pairs for selected sentences.
            - X: TF-IDF feature matrix (scipy.sparse.csr_matrix).
            - scores: Array of TF-IDF scores for all sentences.
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",  # Remove common English stop words (e.g., "the", "is", "on")
        ngram_range=(1, 2),  # Use both unigrams and bigrams for better phrase capture
        max_df=0.95,  # Ignore terms appearing in more than 95% of sentences (too common)
        min_df=1,  # Keep terms that appear in at least 1 sentence
        sublinear_tf=True,  # Apply logarithmic scaling to term frequency (dampen effect of very frequent words)
    )

    # Compute TF-IDF matrix for all sentences
    X = vectorizer.fit_transform(sentences)
    scores = np.asarray(X.mean(axis=1)).flatten()

    # Define adaptive threshold based on percentile
    percentile = 100 * (1 - keep_ratio)
    threshold = np.percentile(scores, percentile)

    if with_log:
        mu, sigma = mean(scores), stdev(scores)
        print(
            f"TF-IDF mean={mu:.4f}, σ={sigma:.4f}, "
            f"threshold (perc={percentile:.0f}%)={threshold:.4f}"
        )

    # Select the most informative sentences
    informative = [
        (s, i) for i, (s, sc) in enumerate(zip(sentences, scores)) if sc >= threshold
    ]

    if with_log:
        print(
            f"Informative sentences: {len(informative)} / {len(sentences)} "
            f"({len(informative)/len(sentences)*100:.1f}%)"
        )

    return informative, X, scores
