"""
Metadata Storage and Search Interfaces

This package contains all the interfaces required for the unified metadata storage
and semantic search service. These interfaces define the contracts between:
- Plugins and the metadata storage system
- The storage system and the search system
- The async job system and the storage/search systems
"""

from .metadata_writer import MetadataWriter
from .metadata_reader import MetadataReader
from .search_service import SearchService
from .embedding_service import EmbeddingService
from .indexing_service import IndexingService

__all__ = [
    'MetadataWriter',
    'MetadataReader',
    'SearchService',
    'EmbeddingService',
    'IndexingService',
]
