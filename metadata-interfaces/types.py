"""
Common types and enums used across all interfaces.

These types provide type safety and clear documentation of what values are expected.
"""

from enum import Enum
from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime


class MediaType(str, Enum):
    """
    Type of media file being processed.
    
    Values:
        IMAGE: Image files (jpg, png, etc.)
        TEXT: Text documents (txt, pdf, docx, etc.)
    """
    IMAGE = "IMAGE"
    TEXT = "TEXT"


class ContentType(str, Enum):
    """
    Type of text content stored in the system.
    
    Values:
        FULL_TEXT: Complete text from a document
        CHUNK: A chunk/segment of a larger document (for embedding)
        CAPTION: Generated caption describing an image
        DESCRIPTION: Generated description of content
    """
    FULL_TEXT = "full_text"
    CHUNK = "chunk"
    CAPTION = "caption"
    DESCRIPTION = "description"


class EmbeddingSource(str, Enum):
    """
    Source type for an embedding.
    
    Values:
        IMAGE: Embedding generated directly from image (e.g., CLIP image encoder)
        TEXT: Embedding generated from text (e.g., sentence transformer)
    """
    IMAGE = "image"
    TEXT = "text"


class ModelType(str, Enum):
    """
    Type of embedding model.
    
    Values:
        CLIP: CLIP model (can embed both images and text)
        SENTENCE_TRANSFORMER: Sentence transformer model (text only)
    """
    CLIP = "clip"
    SENTENCE_TRANSFORMER = "sentence_transformer"


class FileAttribute(TypedDict):
    """
    Represents a single attribute extracted from a file.
    
    Attributes:
        attribute_key: The name/key of the attribute (e.g., "gender", "age_min")
        attribute_value: The string value of the attribute
        plugin_source: Name of the plugin that extracted this attribute
    """
    attribute_key: str
    attribute_value: str
    plugin_source: str


class TextContent(TypedDict):
    """
    Represents text content extracted from or associated with a file.
    
    Attributes:
        text_content: The actual text
        content_type: Type of content (full_text, chunk, caption, etc.)
        chunk_index: For chunked content, the position in the sequence (0-based)
        plugin_source: Name of the plugin that generated this text
    """
    text_content: str
    content_type: ContentType
    chunk_index: Optional[int]
    plugin_source: str


class EmbeddingVector(TypedDict):
    """
    Represents an embedding vector with metadata.
    
    Attributes:
        embedding: The actual vector as a list of floats
        model_name: Name of the model used (must be registered)
        embedding_source: Whether this is from an image or text
        text_content_id: If from text, the ID of the source text content
        plugin_source: Name of the plugin that generated this embedding
    """
    embedding: List[float]
    model_name: str
    embedding_source: EmbeddingSource
    text_content_id: Optional[str]
    plugin_source: str


class SearchFilter(TypedDict, total=False):
    """
    Filter criteria for exact-match attribute searches.
    
    Attributes:
        attribute_key: The attribute to filter on
        attribute_value: Exact value to match (optional)
        value_in: List of values to match (OR logic)
        value_range: Tuple of (min, max) for range queries
    
    Example:
        {"attribute_key": "gender", "attribute_value": "male"}
        {"attribute_key": "age_min", "value_range": (25, 32)}
    """
    attribute_key: str
    attribute_value: Optional[str]
    value_in: Optional[List[str]]
    value_range: Optional[tuple[float, float]]


class SearchResult(TypedDict):
    """
    Result from a search query.
    
    Attributes:
        file_id: UUID of the matching file
        file_path: Path to the file
        media_type: Type of media (IMAGE or TEXT)
        collection_id: UUID of the collection this file belongs to
        collection_name: Name of the collection
        similarity_score: For semantic search, the similarity score (0-1)
        matched_attributes: List of attributes that matched the search
    """
    file_id: str
    file_path: str
    media_type: MediaType
    collection_id: str
    collection_name: str
    similarity_score: Optional[float]
    matched_attributes: List[FileAttribute]


class EmbeddingModel(TypedDict):
    """
    Information about an embedding model.
    
    Attributes:
        model_name: Unique name identifier for the model
        model_type: Type of model (clip or sentence_transformer)
        embedding_dimension: Dimension of vectors this model produces
    """
    model_name: str
    model_type: ModelType
    embedding_dimension: int
