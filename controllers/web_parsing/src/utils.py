from typing import List, Tuple
from nltk.tokenize import sent_tokenize

# import nltk

# nltk.download("punkt")
# nltk.download("punkt_tab")


def split_into_sentences(
    texts: List[str], min_words: int = 4, with_log: bool = False
) -> Tuple[List[str], List[int]]:
    """
    Splits multiple documents into sentences and records which document
    each sentence came from.

    Args:
        texts (List[str]): List of text documents.
        min_words (int): Minimum number of words per sentence to keep.
        with_log (bool): If True, prints the total number of extracted sentences.

    Returns:
        Tuple[List[str], List[int]]: A tuple containing:
            - sentences: all extracted sentences;
            - sentence_doc_ids: document IDs for each sentence.
    """
    sentences, sentence_doc_ids = [], []

    for doc_id, text in enumerate(texts):
        sents = [s.strip() for s in sent_tokenize(text) if len(s.split()) >= min_words]
        sentences.extend(sents)
        sentence_doc_ids.extend([doc_id] * len(sents))

    if with_log:
        print(f"Total sentences extracted: {len(sentences)}")

    return sentences, sentence_doc_ids


def rebuild_docs(
    unique_sentences: List[Tuple[str, int]], sentence_doc_ids: List[int], num_docs: int
) -> List[str]:
    """
    Rebuilds documents from filtered or deduplicated sentences.

    Args:
        unique_sentences (List[Tuple[str, int]]): List of (sentence, sentence_index) pairs.
        sentence_doc_ids (List[int]): Source document ID for each sentence index.
        num_docs (int): Number of original documents.

    Returns:
        List[str]: Reconstructed list of cleaned documents.
    """
    docs = ["" for _ in range(num_docs)]
    for sent, idx in unique_sentences:
        doc_id = sentence_doc_ids[idx]
        docs[doc_id] += sent.strip() + " "
    return docs
