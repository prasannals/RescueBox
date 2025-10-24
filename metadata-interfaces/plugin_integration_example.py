"""
Plugin Integration Example

This file shows how a plugin should integrate with the metadata storage system.
It demonstrates the complete flow from file analysis to metadata storage.
"""

from typing import List, Optional
from metadata_interfaces import MetadataWriter, EmbeddingService
from metadata_interfaces.types import MediaType, FileAttribute, TextContent, ContentType, EmbeddingVector, EmbeddingSource


class ExamplePlugin:
    """
    Example plugin showing integration with metadata storage.
    
    This represents a typical analysis plugin (e.g., age/gender detection,
    image captioning, face recognition) that processes files and stores results.
    """
    
    def __init__(
        self,
        metadata_writer: MetadataWriter,
        embedding_service: Optional[EmbeddingService] = None
    ):
        """
        Initialize the plugin with required services.
        
        Args:
            metadata_writer: Writer interface for storing metadata
            embedding_service: Optional embedder for generating embeddings
        """
        self.writer = metadata_writer
        self.embedder = embedding_service
        self.plugin_name = "example_plugin"
    
    async def process_image(self, file_id: str, image_path: str) -> None:
        """
        Process an image and store all extracted metadata.
        
        This is the main entry point called by the indexing service.
        
        Args:
            file_id: UUID of the file to process
            image_path: Path to the image file
        """
        # Step 1: Run ML model to extract attributes
        attributes = await self._analyze_image(image_path)
        
        # Step 2: Write attributes to storage
        await self.writer.write_attributes(
            file_id=file_id,
            attributes=attributes
        )
        
        # Step 3: Generate description/caption
        caption = await self._generate_caption(image_path)
        
        # Step 4: Write text content
        text_content_id = await self.writer.write_text_content(
            file_id=file_id,
            content={
                "text_content": caption,
                "content_type": ContentType.CAPTION,
                "chunk_index": None,
                "plugin_source": self.plugin_name
            }
        )
        
        # Step 5: Generate and store embedding (if embedder available)
        if self.embedder:
            # Embed the caption text
            caption_embedding = await self.embedder.embed_text(
                text=caption,
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            await self.writer.write_embedding(
                file_id=file_id,
                embedding={
                    "embedding": caption_embedding,
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "embedding_source": EmbeddingSource.TEXT,
                    "text_content_id": text_content_id,
                    "plugin_source": self.plugin_name
                }
            )
            
            # Embed the image directly
            image_embedding = await self.embedder.embed_image(
                image_path=image_path,
                model_name="openai/clip-vit-base-patch32"
            )
            
            await self.writer.write_embedding(
                file_id=file_id,
                embedding={
                    "embedding": image_embedding,
                    "model_name": "openai/clip-vit-base-patch32",
                    "embedding_source": EmbeddingSource.IMAGE,
                    "text_content_id": None,
                    "plugin_source": self.plugin_name
                }
            )
        
        # Step 6: Mark file as indexed
        await self.writer.update_file_indexed_timestamp(file_id)
    
    async def _analyze_image(self, image_path: str) -> List[FileAttribute]:
        """
        Run ML model to extract attributes from image.
        
        This is plugin-specific logic - could be age/gender detection,
        object detection, scene classification, etc.
        
        Returns:
            List of FileAttribute dictionaries
        """
        # Placeholder for actual ML inference
        # In real implementation, this would:
        # 1. Load the image
        # 2. Run inference
        # 3. Parse results
        # 4. Return structured attributes
        
        return [
            {
                "attribute_key": "gender",
                "attribute_value": "male",
                "plugin_source": self.plugin_name
            },
            {
                "attribute_key": "age_min",
                "attribute_value": "25",
                "plugin_source": self.plugin_name
            },
            {
                "attribute_key": "age_max",
                "attribute_value": "32",
                "plugin_source": self.plugin_name
            },
            {
                "attribute_key": "scene_type",
                "attribute_value": "outdoor",
                "plugin_source": self.plugin_name
            }
        ]
    
    async def _generate_caption(self, image_path: str) -> str:
        """
        Generate a natural language caption for the image.
        
        This would use an image captioning model (BLIP, etc.).
        
        Returns:
            Caption text
        """
        # Placeholder for actual caption generation
        return "A person skiing down a snowy mountain slope"


class TextDocumentPlugin:
    """
    Example plugin for processing text documents.
    
    Shows how to handle chunking and text embeddings.
    """
    
    def __init__(
        self,
        metadata_writer: MetadataWriter,
        embedding_service: EmbeddingService
    ):
        self.writer = metadata_writer
        self.embedder = embedding_service
        self.plugin_name = "text_document_plugin"
        self.chunk_size = 512  # characters per chunk
    
    async def process_document(self, file_id: str, document_path: str) -> None:
        """
        Process a text document: extract text, chunk it, embed chunks.
        
        Args:
            file_id: UUID of the file
            document_path: Path to the text document
        """
        # Step 1: Extract text (might involve PDF parsing, OCR, etc.)
        full_text = await self._extract_text(document_path)
        
        # Step 2: Write full text
        await self.writer.write_text_content(
            file_id=file_id,
            content={
                "text_content": full_text,
                "content_type": ContentType.FULL_TEXT,
                "chunk_index": None,
                "plugin_source": self.plugin_name
            }
        )
        
        # Step 3: Chunk the text
        chunks = self._chunk_text(full_text)
        
        # Step 4: Write chunks and generate embeddings
        chunk_texts = []
        for i, chunk in enumerate(chunks):
            text_content_id = await self.writer.write_text_content(
                file_id=file_id,
                content={
                    "text_content": chunk,
                    "content_type": ContentType.CHUNK,
                    "chunk_index": i,
                    "plugin_source": self.plugin_name
                }
            )
            chunk_texts.append(chunk)
        
        # Step 5: Batch embed all chunks
        embeddings = await self.embedder.embed_text_batch(
            texts=chunk_texts,
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Step 6: Write embeddings
        # Note: In production, you'd track text_content_ids from step 4
        for i, embedding in enumerate(embeddings):
            await self.writer.write_embedding(
                file_id=file_id,
                embedding={
                    "embedding": embedding,
                    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                    "embedding_source": EmbeddingSource.TEXT,
                    "text_content_id": f"chunk-{i}",  # Use actual ID from step 4
                    "plugin_source": self.plugin_name
                }
            )
        
        # Step 7: Mark as indexed
        await self.writer.update_file_indexed_timestamp(file_id)
    
    async def _extract_text(self, document_path: str) -> str:
        """Extract text from document (PDF, DOCX, etc.)."""
        # Placeholder - would use pypdf, python-docx, etc.
        return "This is the extracted document text..."
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks for embedding.
        
        In production, use LangChain's RecursiveCharacterTextSplitter
        or similar for smarter chunking.
        """
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunks.append(text[i:i + self.chunk_size])
        return chunks


# Example usage in an indexing worker
async def example_indexing_flow():
    """
    Shows how plugins are called during indexing.
    
    This would be inside the IndexingService implementation.
    """
    from metadata_interfaces import MetadataWriter, EmbeddingService, IndexingService
    
    # Initialize services (with real implementations)
    writer = MetadataWriter(...)  # Pass database connection
    embedder = EmbeddingService(...)  # Pass model config
    indexer = IndexingService(...)  # Pass Celery config
    
    # Create plugin instances
    image_plugin = ExamplePlugin(
        metadata_writer=writer,
        embedding_service=embedder
    )
    
    text_plugin = TextDocumentPlugin(
        metadata_writer=writer,
        embedding_service=embedder
    )
    
    # Register collection
    collection_id = await writer.register_collection("case_2024_001")
    
    # Register files
    image_file_id = await writer.register_file(
        collection_id=collection_id,
        file_path="/data/images/photo001.jpg",
        media_type=MediaType.IMAGE
    )
    
    text_file_id = await writer.register_file(
        collection_id=collection_id,
        file_path="/data/docs/report.pdf",
        media_type=MediaType.TEXT
    )
    
    # Process files with plugins
    await image_plugin.process_image(
        file_id=image_file_id,
        image_path="/data/images/photo001.jpg"
    )
    
    await text_plugin.process_document(
        file_id=text_file_id,
        document_path="/data/docs/report.pdf"
    )
    
    print("Indexing complete!")


# Example search usage
async def example_search_flow():
    """
    Shows how to use the search interfaces.
    """
    from metadata_interfaces import SearchService, MetadataReader
    from metadata_interfaces.types import MediaType, EmbeddingSource
    
    search = SearchService(...)  # Initialize with database
    reader = MetadataReader(...)  # Initialize with database
    
    # Use case: "find all images of males aged 25-32 in mountain resort like areas"
    results = await search.combined_search(
        attribute_filters=[
            {"attribute_key": "gender", "attribute_value": "male"},
            {"attribute_key": "age_min", "value_range": (25, 32)},
            {"attribute_key": "age_max", "value_range": (25, 32)}
        ],
        semantic_query="mountain resort skiing snow",
        combine_strategy="FILTER_THEN_SEARCH",
        filter_operator="AND",
        media_types=[MediaType.IMAGE],
        embedding_source=EmbeddingSource.IMAGE,
        top_k=50,
        similarity_threshold=0.7
    )
    
    # Display results
    print(f"Found {len(results)} matching images:")
    for result in results:
        print(f"\nFile: {result['file_path']}")
        print(f"Similarity: {result['similarity_score']:.2f}")
        print(f"Collection: {result['collection_name']}")
        
        # Get detailed attributes
        attributes = await reader.get_file_attributes(result['file_id'])
        print("Attributes:")
        for attr in attributes:
            print(f"  {attr['attribute_key']}: {attr['attribute_value']}")
        
        # Get caption if available
        text_content = await reader.get_file_text_content(
            result['file_id'],
            content_type="caption"
        )
        if text_content:
            print(f"Caption: {text_content[0]['text_content']}")
