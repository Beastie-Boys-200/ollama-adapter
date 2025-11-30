from typing import List, Optional
from pydantic import BaseModel


class Article(BaseModel):
    link: str
    text: str


class SearchAndExtractRequest(BaseModel):
    query: str
    count: int = 10
    with_log: bool = False


class ExtractRequest(BaseModel):
    links: List[str]
    drop_tags: Optional[List[str]] = None
    with_log: bool = False


class SemanticCleanRequest(BaseModel):
    texts: List[str]
    keep_ratio: float = 0.7
    similarity_threshold: float = 0.9
    with_log: bool = False
