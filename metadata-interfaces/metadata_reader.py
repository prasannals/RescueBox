"""
MetadataReader Interface

This interface is used to read metadata from the storage system without performing
searches. It's primarily used for:
- Displaying file details and associated metadata
- Debugging and inspection
- Building UIs that show what metadata exists for a file
- Retrieving specific attributes by key

Usage Example:
    reader = MetadataReader(...)
    
    # Get all attributes for a file
    attributes = await reader.get_file_attributes(file_id="abc-123")
    
    # Get specific attribute value
    gender = await reader.get_attribute_value(
        file_id="abc-123",
        attribute_key="gender"
    )
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from .types import FileAttribute, TextContent, MediaType


class MetadataReader(ABC):
    """
    Interface for reading metadata from the storage system.
    
    This provides read-only access to metadata without performing searches.
    All methods are async to support efficient database I/O.
    """
    
    @abstractmethod
    async def get_file_info(
        self,
        file_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get basic information about a file.
        
        Purpose:
            Retrieves core file metadata including path, type, collection,
            and indexing status.
        
        Args:
            file_id: UUID of the file
        
        Returns:
            Dictionary containing:
                - file_id: UUID of the file
                - collection_id: UUID of the parent collection
                - collection_name: Name of the parent collection
                - file_path: Path to the file
                - media_type: Type of media (IMAGE or TEXT)
                - created_at: When file was registered
                - indexed_at: When file was last indexed (or None)
            Returns None if file doesn't exist.
        
        Example:
            info = await reader.get_file_info(file_id="abc-123")
            if info:
                print(f"File: {info['file_path']}")
                print(f"Type: {info['media_type']}")
                print(f"Last indexed: {info['indexed_at']}")
        """
        pass
    
    @abstractmethod
    async def get_file_attributes(
        self,
        file_id: str,
        plugin_source: Optional[str] = None
    ) -> List[FileAttribute]:
        """
        Get all attributes for a file.
        
        Purpose:
            Retrieves all exact-match attributes associated with a file,
            optionally filtered by plugin source.
        
        Args:
            file_id: UUID of the file
            plugin_source: Optional filter by plugin name. If provided, only
                          returns attributes from that plugin.
        
        Returns:
            List of FileAttribute dictionaries, each containing:
                - attribute_key: Name of the attribute
                - attribute_value: Value of the attribute
                - plugin_source: Plugin that created this attribute
        
        Example:
            # Get all attributes
            all_attrs = await reader.get_file_attributes(file_id="abc-123")
            
            # Get only age/gender detection attributes
            age_gender_attrs = await reader.get_file_attributes(
                file_id="abc-123",
                plugin_source="age_and_gender_detection"
            )
            
            for attr in age_gender_attrs:
                print(f"{attr['attribute_key']}: {attr['attribute_value']}")
        """
        pass
    
    @abstractmethod
    async def get_attribute_value(
        self,
        file_id: str,
        attribute_key: str
    ) -> Optional[str]:
        """
        Get a specific attribute value for a file.
        
        Purpose:
            Quick lookup of a single attribute value by key.
        
        Args:
            file_id: UUID of the file
            attribute_key: Name of the attribute to retrieve
        
        Returns:
            The attribute value as a string, or None if not found.
            If multiple values exist for the same key (from different plugins),
            returns the most recently created one.
        
        Example:
            gender = await reader.get_attribute_value(
                file_id="abc-123",
                attribute_key="gender"
            )
            if gender:
                print(f"Detected gender: {gender}")
        """
        pass
    
    @abstractmethod
    async def get_file_text_content(
        self,
        file_id: str,
        content_type: Optional[str] = None
    ) -> List[TextContent]:
        """
        Get text content associated with a file.
        
        Purpose:
            Retrieves text content (captions, descriptions, chunks) for a file,
            optionally filtered by content type.
        
        Args:
            file_id: UUID of the file
            content_type: Optional filter (full_text, chunk, caption, description)
        
        Returns:
            List of TextContent dictionaries, each containing:
                - text_content: The actual text
                - content_type: Type of content
                - chunk_index: Position in sequence (for chunks) or None
                - plugin_source: Plugin that generated this
            Ordered by chunk_index for chunked content.
        
        Example:
            # Get all text content
            all_text = await reader.get_file_text_content(file_id="abc-123")
            
            # Get only captions
            captions = await reader.get_file_text_content(
                file_id="abc-123",
                content_type="caption"
            )
            
            for caption in captions:
                print(f"Caption: {caption['text_content']}")
                print(f"From: {caption['plugin_source']}")
        """
        pass
    
    @abstractmethod
    async def get_collection_files(
        self,
        collection_id: str,
        media_type: Optional[MediaType] = None,
        indexed_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get all files in a collection.
        
        Purpose:
            Lists all files belonging to a collection, with optional filtering
            by media type and indexing status.
        
        Args:
            collection_id: UUID of the collection
            media_type: Optional filter by media type (IMAGE or TEXT)
            indexed_only: If True, only returns files that have been indexed
        
        Returns:
            List of file info dictionaries (same format as get_file_info).
        
        Example:
            # Get all files in collection
            all_files = await reader.get_collection_files(
                collection_id="xyz-789"
            )
            
            # Get only indexed images
            indexed_images = await reader.get_collection_files(
                collection_id="xyz-789",
                media_type=MediaType.IMAGE,
                indexed_only=True
            )
            
            print(f"Found {len(indexed_images)} indexed images")
        """
        pass
    
    @abstractmethod
    async def get_files_by_path_pattern(
        self,
        path_pattern: str
    ) -> List[Dict[str, Any]]:
        """
        Find files matching a path pattern.
        
        Purpose:
            Searches for files by path pattern (useful for finding files in
            a specific directory or with specific extensions).
        
        Args:
            path_pattern: SQL LIKE pattern (use % as wildcard)
                         Example: "/data/images/%.jpg"
        
        Returns:
            List of file info dictionaries.
        
        Example:
            # Find all JPG files
            jpg_files = await reader.get_files_by_path_pattern("%.jpg")
            
            # Find files in specific directory
            dir_files = await reader.get_files_by_path_pattern("/data/case1/%")
        """
        pass
    
    @abstractmethod
    async def get_attribute_statistics(
        self,
        attribute_key: str,
        collection_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get statistics about an attribute across files.
        
        Purpose:
            Provides aggregate statistics for an attribute, useful for
            understanding data distribution and building UI controls.
        
        Args:
            attribute_key: Name of the attribute to analyze
            collection_id: Optional filter to specific collection
        
        Returns:
            Dictionary containing:
                - attribute_key: The attribute name
                - total_files: Number of files with this attribute
                - unique_values: List of unique values
                - value_counts: Dictionary mapping value -> count
                - numeric_stats: If numeric, includes min/max/avg (or None)
        
        Example:
            stats = await reader.get_attribute_statistics(
                attribute_key="gender",
                collection_id="xyz-789"
            )
            
            print(f"Gender distribution:")
            for value, count in stats['value_counts'].items():
                print(f"  {value}: {count}")
        """
        pass
