"""
SearchService Interface

This is the main interface for performing searches across the metadata store.
It supports:
- Exact-match filtering on attributes
- Semantic/similarity search using embeddings
- Combined searches (exact match + semantic)

Usage Example:
    search = SearchService(...)
    
    # Exact match search
    results = await search.exact_match_search(
        filters=[
            {"attribute_key": "gender", "attribute_value": "male"},
            {"attribute_key": "age_min", "value_range": (25, 32)}
        ],
        operator="AND"
    )
    
    # Semantic search
    results = await search.semantic_search(
        query_text="mountain resort skiing",
        media_types=[MediaType.IMAGE],
        top_k=50
    )
    
    # Combined search
    results = await search.combined_search(
        attribute_filters=[
            {"attribute_key": "gender", "attribute_value": "male"}
        ],
        semantic_query="mountain resort",
        combine_strategy="AND"
    )
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from .types import SearchFilter, SearchResult, MediaType, EmbeddingSource


class SearchService(ABC):
    """
    Interface for searching files based on metadata and embeddings.
    
    This is the primary search interface that enables the core use case:
    "find all images of males aged 25-32 in mountain resort like areas"
    """
    
    @abstractmethod
    async def exact_match_search(
        self,
        filters: List[SearchFilter],
        operator: str = "AND",
        collection_id: Optional[str] = None,
        media_type: Optional[MediaType] = None
    ) -> List[SearchResult]:
        """
        Search files using exact-match attribute filters.
        
        Purpose:
            Finds files matching specific attribute criteria. Supports exact
            matching, list matching (IN), and range queries. Multiple filters
            can be combined with AND/OR logic.
        
        Args:
            filters: List of filter dictionaries, each containing:
                - attribute_key: The attribute to filter on
                - attribute_value: Exact value to match (optional)
                - value_in: List of values (OR within list) (optional)
                - value_range: Tuple of (min, max) for numeric range (optional)
            operator: How to combine filters - "AND" or "OR"
            collection_id: Optional filter to specific collection
            media_type: Optional filter by media type (IMAGE or TEXT)
        
        Returns:
            List of SearchResult dictionaries, each containing:
                - file_id: UUID of the matching file
                - file_path: Path to the file
                - media_type: Type of media
                - collection_id: UUID of the collection
                - collection_name: Name of the collection
                - similarity_score: None (not applicable for exact match)
                - matched_attributes: List of attributes that matched
        
        Examples:
            # Find all males
            results = await search.exact_match_search(
                filters=[
                    {"attribute_key": "gender", "attribute_value": "male"}
                ]
            )
            
            # Find males OR females aged 25-32
            results = await search.exact_match_search(
                filters=[
                    {"attribute_key": "gender", "value_in": ["male", "female"]},
                    {"attribute_key": "age_min", "value_range": (25, 32)}
                ],
                operator="AND"
            )
            
            # Find in specific collection
            results = await search.exact_match_search(
                filters=[{"attribute_key": "scene_type", "attribute_value": "outdoor"}],
                collection_id="xyz-789",
                media_type=MediaType.IMAGE
            )
        
        Notes:
            - For range queries, the system will attempt to cast attribute_value
              to numeric. Ensure attributes are stored as numbers.
            - Empty filters list returns all files (subject to collection/type filters)
            - Results are unordered; use semantic_search for relevance ranking
        """
        pass
    
    @abstractmethod
    async def semantic_search(
        self,
        query_text: str,
        media_types: Optional[List[MediaType]] = None,
        embedding_source: Optional[EmbeddingSource] = None,
        model_name: Optional[str] = None,
        top_k: int = 50,
        similarity_threshold: float = 0.0,
        collection_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search files using semantic similarity.
        
        Purpose:
            Finds files semantically similar to a query text. The query is
            embedded using the specified model, then nearest neighbors are
            found using vector similarity (cosine distance).
        
        Args:
            query_text: Natural language query to search for
            media_types: Optional list of media types to search
            embedding_source: Optional filter (image or text embeddings)
            model_name: Optional specific model to use for embedding the query.
                       If None, uses the default active model for the source type.
            top_k: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0-1). Results below
                                 this threshold are excluded.
            collection_id: Optional filter to specific collection
        
        Returns:
            List of SearchResult dictionaries, ordered by similarity (best first).
            similarity_score will be populated with cosine similarity (0-1).
        
        Examples:
            # Find images of mountains
            results = await search.semantic_search(
                query_text="mountain resort skiing",
                media_types=[MediaType.IMAGE],
                embedding_source=EmbeddingSource.IMAGE,
                top_k=50,
                similarity_threshold=0.7
            )
            
            for result in results:
                print(f"File: {result['file_path']}")
                print(f"Similarity: {result['similarity_score']:.2f}")
            
            # Search text documents
            results = await search.semantic_search(
                query_text="forensic analysis procedures",
                media_types=[MediaType.TEXT],
                embedding_source=EmbeddingSource.TEXT,
                top_k=20
            )
        
        Notes:
            - The query_text is embedded on-the-fly using the specified model
            - Higher top_k values may be slower but more comprehensive
            - similarity_threshold helps filter out poor matches
            - Results are always ordered by similarity (descending)
        """
        pass
    
    @abstractmethod
    async def combined_search(
        self,
        attribute_filters: List[SearchFilter],
        semantic_query: str,
        combine_strategy: str = "AND",
        filter_operator: str = "AND",
        media_types: Optional[List[MediaType]] = None,
        embedding_source: Optional[EmbeddingSource] = None,
        top_k: int = 50,
        similarity_threshold: float = 0.0,
        collection_id: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search using both exact-match filters and semantic similarity.
        
        Purpose:
            Combines attribute filtering with semantic search to support
            queries like "find males aged 25-32 in mountain resort scenes".
            This is the most powerful search method.
        
        Args:
            attribute_filters: List of exact-match filters (same as exact_match_search)
            semantic_query: Natural language query for semantic matching
            combine_strategy: How to combine exact and semantic results:
                - "AND": Files must match BOTH filters AND semantic query
                - "OR": Files matching EITHER filters OR semantic query
                - "FILTER_THEN_SEARCH": Apply filters first, then semantic search
                                       within results (most efficient)
                - "SEARCH_THEN_FILTER": Semantic search first, then filter results
            filter_operator: How to combine attribute filters - "AND" or "OR"
            media_types: Optional list of media types
            embedding_source: Optional embedding source filter
            top_k: Maximum number of results (applies to semantic portion)
            similarity_threshold: Minimum similarity for semantic matches
            collection_id: Optional collection filter
        
        Returns:
            List of SearchResult dictionaries. If combine_strategy involves
            semantic search, results are ordered by similarity.
        
        Examples:
            # Find males aged 25-32 in mountain scenes
            results = await search.combined_search(
                attribute_filters=[
                    {"attribute_key": "gender", "attribute_value": "male"},
                    {"attribute_key": "age_min", "value_range": (25, 32)}
                ],
                semantic_query="mountain resort skiing",
                combine_strategy="FILTER_THEN_SEARCH",
                filter_operator="AND",
                media_types=[MediaType.IMAGE],
                embedding_source=EmbeddingSource.IMAGE,
                top_k=50,
                similarity_threshold=0.7
            )
            
            # Find outdoor scenes OR images with people
            results = await search.combined_search(
                attribute_filters=[
                    {"attribute_key": "scene_type", "attribute_value": "outdoor"}
                ],
                semantic_query="people standing walking",
                combine_strategy="OR",
                top_k=100
            )
        
        Notes:
            - "FILTER_THEN_SEARCH" is most efficient when filters are selective
            - "SEARCH_THEN_FILTER" is better when semantic query is very specific
            - "AND" provides strictest matching, "OR" provides highest recall
            - Results include matched_attributes from the exact-match portion
        """
        pass
    
    @abstractmethod
    async def search_by_file_similarity(
        self,
        file_id: str,
        embedding_source: Optional[EmbeddingSource] = None,
        top_k: int = 50,
        similarity_threshold: float = 0.0,
        collection_id: Optional[str] = None,
        exclude_self: bool = True
    ) -> List[SearchResult]:
        """
        Find files similar to a given file.
        
        Purpose:
            Given a file, finds other files with similar embeddings. Useful
            for "find more like this" functionality.
        
        Args:
            file_id: UUID of the reference file
            embedding_source: Which embeddings to use (image or text)
            top_k: Maximum number of results
            similarity_threshold: Minimum similarity score
            collection_id: Optional filter to specific collection
            exclude_self: If True, excludes the query file from results
        
        Returns:
            List of SearchResult dictionaries, ordered by similarity.
        
        Example:
            # Find images similar to this one
            similar = await search.search_by_file_similarity(
                file_id="abc-123",
                embedding_source=EmbeddingSource.IMAGE,
                top_k=20,
                exclude_self=True
            )
            
            print(f"Found {len(similar)} similar images")
        
        Notes:
            - Uses the file's existing embeddings (doesn't re-embed)
            - If file has multiple embeddings, uses the most recent one
            - exclude_self=True is typical to avoid returning the query file
        """
        pass
    
    @abstractmethod
    async def count_results(
        self,
        filters: Optional[List[SearchFilter]] = None,
        semantic_query: Optional[str] = None,
        combine_strategy: str = "AND",
        collection_id: Optional[str] = None,
        media_type: Optional[MediaType] = None
    ) -> int:
        """
        Count how many files match search criteria without retrieving them.
        
        Purpose:
            Efficiently counts matches for pagination or UI display without
            the overhead of retrieving full results.
        
        Args:
            filters: Optional exact-match filters
            semantic_query: Optional semantic query
            combine_strategy: How to combine if both provided
            collection_id: Optional collection filter
            media_type: Optional media type filter
        
        Returns:
            Integer count of matching files.
        
        Example:
            # Count males in collection
            count = await search.count_results(
                filters=[{"attribute_key": "gender", "attribute_value": "male"}],
                collection_id="xyz-789"
            )
            print(f"Found {count} males in collection")
        
        Notes:
            - Much faster than len(search_results) for large result sets
            - Useful for pagination (showing "Page 1 of 10")
        """
        pass
