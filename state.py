from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import numpy as np
import faiss

class IngestionState(BaseModel):
    file_path: str
    elements: Optional[List] = None
    text_chunks: Optional[List[Dict]] = None
    image_chunks: Optional[List[Dict]] = None
    table_chunks: Optional[List[Dict]] = None
    all_chunks: Optional[List[Dict]] = None
    has_text: bool = False
    has_table: bool = False
    has_visual: bool = False
    embeddings: Optional[np.ndarray] = None
    metadatas: Optional[List[Dict]] = None
    index_folder: Optional[str] = 'vector_store'
    faiss_index: Optional[faiss.IndexFlatL2] = None
    metadata_path: Optional[str] = None
    vlm_prompt: Optional[str] = None
    vlm_model: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    error: Optional[str] = None
    texts: Optional[List[str]] = []

    class Config:
        arbitrary_types_allowed = True


class QueryState(BaseModel):
    query: str
    index_path: str = "vector_store/faiss.index"
    metadata_path: str = "vector_store/metadatas.pkl"
    embedding_model: str = "all-MiniLM-L6-v2"
    plot_requested: bool = False
    table_requested: bool = False
    query_embedding: Optional[np.ndarray] = None
    search_results: Optional[Dict] = None
    retrieved_chunks: Optional[List[Dict]] = None
    highlighted_pages: Optional[List] = None
    generated_plots: Optional[List] = None
    query_modality: Optional[str] = "text_only"
    retrieved_dataframe: Optional[Any] = None
    answer: Optional[str] = None
    loaded_metadatas: Optional[Dict] = None
    faiss_index: Optional[faiss.IndexFlatL2] = None
    retries: int = 0
    max_retries: int = 3
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True