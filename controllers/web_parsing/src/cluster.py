from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering


def cluster_similar_sentences(
    informative: List[Tuple[str, int]],
    X,
    similarity_threshold: float = 0.9,
    with_log: bool = False,
) -> List[Tuple[str, int]]:
    """
    Clusters semantically similar sentences using cosine similarity
    and returns one representative sentence per cluster.

    Args:
        informative (List[Tuple[str, int]]):
            List of (sentence_text, global_sentence_index) pairs selected
            as informative by the TF-IDF filtering step.
        X:
            TF-IDF feature matrix (scipy.sparse.csr_matrix) for all sentences.
        similarity_threshold (float):
            Cosine similarity threshold for clustering (typical range 0.85–0.95).
            Higher values mean stricter merging — only nearly identical
            sentences will be grouped.
        with_log (bool):
            If True, prints clustering statistics and progress.

    Returns:
        List[Tuple[str, int]]:
            A list of unique representative sentences, one per cluster.
    """
    selected_sentences = [s for s, _ in informative]
    selected_vectors = X[[i for _, i in informative]]

    if len(selected_sentences) <= 1:
        if with_log:
            print("Too few sentences for clustering — skipping.")
        return informative

    # Compute pairwise cosine distances
    sim_matrix = cosine_similarity(selected_vectors)
    dist_matrix = 1 - sim_matrix  # Convert similarity to distance

    # Hierarchical agglomerative clustering based on cosine distance
    clustering = AgglomerativeClustering(
        n_clusters=None,  # Automatically determine the number of clusters
        metric="precomputed",  # Use our custom distance matrix (1 - cosine_similarity)
        linkage="complete",  # Merge clusters based on the farthest pairwise distance
        distance_threshold=1
        - similarity_threshold,  # Stop merging when similarity falls below the threshold
    )
    labels = clustering.fit_predict(dist_matrix)

    # Group sentences by cluster label
    clusters = {}
    for label, (sentence, i) in zip(labels, informative):
        clusters.setdefault(label, []).append((sentence, i))

    if with_log:
        print(f"Clusters after deduplication: {len(clusters)}")

    # Select the longest sentence in each cluster as its representative
    unique = [max(sents, key=lambda x: len(x[0])) for sents in clusters.values()]

    if with_log:
        print(
            f"Unique sentences kept: {len(unique)} / {len(selected_sentences)} "
            f"({len(unique) / len(selected_sentences) * 100:.1f}%)"
        )

    return unique
