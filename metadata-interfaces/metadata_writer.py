"""
MetadataWriter Interface

This interface is used by plugins to write metadata (attributes, text content, embeddings)
to the unified metadata storage. Each plugin should receive an instance of this interface
and use it to persist their analysis results.

Usage Example:
    writer = MetadataWriter(...)
    
    # Write attributes extracted by the plugin
    await writer.write_attributes(
        file_id="abc-123",
        attributes=[
            {"attribute_key": "gender", "attribute_value": "male", "plugin_source": "age_gender_detection"},
            {"attribute_key": "age_min", "attribute_value": "25", "plugin_source": "age_gender_detection"},
        ]
    )
    
    # Write text description for embedding
    await writer.write_text_content(
        file_id="abc-123",
        content={
            "text_content": "A person standing in a mountain resort",
            "content_type": ContentType.DESCRIPTION,
            "chunk_index": None,
            "plugin_source": "image_summary"
        }
    )
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from .types import FileAttribute, TextContent, EmbeddingVector, MediaType


class MetadataWriter(ABC):
    """
    Interface for writing metadata to the storage system.
    
    This is the primary interface that plugins use to persist their results.
    All methods are async to support efficient bulk operations and database I/O.
    """
    
    @abstractmethod
    async def register_collection(
        self,
        collection_name: str
    ) -> str:
        """
        Register a new media collection (e.g., a case folder).
        
        Purpose:
            Creates a logical grouping for media files. This should be called
            before adding files to a new collection. If the collection already
            exists, returns its ID.
        
        Args:
            collection_name: Human-readable name for the collection
                           (e.g., folder name, case ID)
        
        Returns:
            collection_id: UUID of the created or existing collection
        
        Example:
            collection_id = await writer.register_collection("case_2024_001")
        """
        pass
    
    @abstractmethod
    async def register_file(
        self,
        collection_id: str,
        file_path: str,
        media_type: MediaType
    ) -> str:
        """
        Register a new file in the system.
        
        Purpose:
            Adds a file to the metadata store. This should be called before
            any plugin processes the file. If the file already exists, returns
            its existing ID.
        
        Args:
            collection_id: UUID of the collection this file belongs to
            file_path: Absolute or relative path to the file
            media_type: Type of media (IMAGE or TEXT)
        
        Returns:
            file_id: UUID of the registered file
        
        Example:
            file_id = await writer.register_file(
                collection_id="xyz-789",
                file_path="/data/images/photo001.jpg",
                media_type=MediaType.IMAGE
            )
        """
        pass
    
    @abstractmethod
    async def write_attributes(
        self,
        file_id: str,
        attributes: List[FileAttribute]
    ) -> None:
        """
        Write exact-match attributes for a file.
        
        Purpose:
            Stores structured metadata that can be used for exact-match and
            range queries. Examples: gender, age range, detected objects, etc.
            Multiple attributes can be written in a single call for efficiency.
        
        Args:
            file_id: UUID of the file these attributes belong to
            attributes: List of attribute dictionaries, each containing:
                - attribute_key: Name of the attribute (e.g., "gender")
                - attribute_value: String value (e.g., "male")
                - plugin_source: Name of the plugin that extracted this
        
        Returns:
            None
        
        Example:
            await writer.write_attributes(
                file_id="abc-123",
                attributes=[
                    {
                        "attribute_key": "gender",
                        "attribute_value": "male",
                        "plugin_source": "age_and_gender_detection"
                    },
                    {
                        "attribute_key": "age_min",
                        "attribute_value": "25",
                        "plugin_source": "age_and_gender_detection"
                    },
                    {
                        "attribute_key": "age_max",
                        "attribute_value": "32",
                        "plugin_source": "age_and_gender_detection"
                    }
                ]
            )
        
        Notes:
            - For range queries (age, height, etc.), use separate attributes
              for min and max (e.g., "age_min" and "age_max")
            - Attribute values are stored as strings; numeric conversions
              happen at query time
            - Existing attributes for the same file/key combination will be
              overwritten (last write wins)
        """
        pass
    
    @abstractmethod
    async def write_text_content(
        self,
        file_id: str,
        content: TextContent
    ) -> str:
        """
        Write text content associated with a file.
        
        Purpose:
            Stores text that will be embedded for semantic search. This includes
            image captions, document chunks, OCR results, or descriptions.
        
        Args:
            file_id: UUID of the file this text is associated with
            content: Dictionary containing:
                - text_content: The actual text
                - content_type: Type (full_text, chunk, caption, description)
                - chunk_index: Position in sequence (for chunked text), or None
                - plugin_source: Name of the plugin that generated this
        
        Returns:
            text_content_id: UUID of the stored text content
        
        Example:
            # Image caption
            text_id = await writer.write_text_content(
                file_id="abc-123",
                content={
                    "text_content": "A person skiing in a mountain resort",
                    "content_type": ContentType.CAPTION,
                    "chunk_index": None,
                    "plugin_source": "image_summary"
                }
            )
            
            # Document chunk
            text_id = await writer.write_text_content(
                file_id="def-456",
                content={
                    "text_content": "This section discusses forensic procedures...",
                    "content_type": ContentType.CHUNK,
                    "chunk_index": 0,
                    "plugin_source": "doc_parser"
                }
            )
        
        Notes:
            - For long documents, split into chunks and call this method for each
            - chunk_index should be sequential starting from 0
            - The returned text_content_id should be passed when writing embeddings
        """
        pass
    
    @abstractmethod
    async def write_embedding(
        self,
        file_id: str,
        embedding: EmbeddingVector
    ) -> None:
        """
        Write an embedding vector for semantic search.
        
        Purpose:
            Stores embedding vectors that enable semantic/similarity search.
            These can be image embeddings (CLIP) or text embeddings (sentence
            transformers).
        
        Args:
            file_id: UUID of the file this embedding represents
            embedding: Dictionary containing:
                - embedding: List of floats (the vector)
                - model_name: Name of the embedding model (must be registered)
                - embedding_source: Whether from image or text
                - text_content_id: If from text, the source text content ID
                - plugin_source: Name of the plugin that generated this
        
        Returns:
            None
        
        Example:
            # Image embedding
            await writer.write_embedding(
                file_id="abc-123",
                embedding={
                    "embedding": [0.123, -0.456, 0.789, ...],  # 512-dim vector
                    "model_name": "openai/clip-vit-base-patch32",
                    "embedding_source": EmbeddingSource.IMAGE,
                    "text_content_id": None,
                    "plugin_source": "clip_embedder"
                }
            )
            
            # Text embedding
            await writer.write_embedding(
                file_id="abc-123",
                embedding={
                    "embedding": [0.321, 0.654, -0.987, ...],  # 384-dim vector
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "embedding_source": EmbeddingSource.TEXT,
                    "text_content_id": "text-id-789",
                    "plugin_source": "text_embedder"
                }
            )
        
        Notes:
            - The embedding dimension must match the registered model's dimension
            - For text embeddings, text_content_id should reference previously
              written text content
            - Multiple embeddings per file are allowed (e.g., image + caption)
        """
        pass
    
    @abstractmethod
    async def update_file_indexed_timestamp(
        self,
        file_id: str
    ) -> None:
        """
        Mark a file as successfully indexed.
        
        Purpose:
            Updates the indexed_at timestamp for a file after successful processing.
            This helps track which files have been processed and when.
        
        Args:
            file_id: UUID of the file that was indexed
        
        Returns:
            None
        
        Example:
            # After all plugin processing is complete
            await writer.update_file_indexed_timestamp(file_id="abc-123")
        
        Notes:
            - Typically called at the end of a successful indexing job
            - Can be used to identify stale or unindexed files
        """
        pass
