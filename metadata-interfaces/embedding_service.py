"""
EmbeddingService Interface

This interface handles the generation of embeddings from raw inputs (images, text).
It abstracts away the ML models and provides a clean API for:
- Registering embedding models
- Generating embeddings from images
- Generating embeddings from text
- Managing model lifecycle

Usage Example:
    embedder = EmbeddingService(...)
    
    # Register a model
    await embedder.register_model(
        model_name="openai/clip-vit-base-patch32",
        model_type=ModelType.CLIP,
        embedding_dimension=512
    )
    
    # Embed an image
    embedding = await embedder.embed_image(
        image_path="/data/photo.jpg",
        model_name="openai/clip-vit-base-patch32"
    )
    
    # Embed text
    embedding = await embedder.embed_text(
        text="A person skiing in mountains",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from .types import EmbeddingModel, ModelType


class EmbeddingService(ABC):
    """
    Interface for generating embeddings from images and text.
    
    This service manages embedding models and provides methods to generate
    embedding vectors. It abstracts the ML implementation details from the
    rest of the system.
    """
    
    @abstractmethod
    async def register_model(
        self,
        model_name: str,
        model_type: ModelType,
        embedding_dimension: int
    ) -> str:
        """
        Register an embedding model in the system.
        
        Purpose:
            Registers a new embedding model so it can be used for generating
            embeddings. This creates a record in the database and may also
            load/initialize the model.
        
        Args:
            model_name: Full identifier for the model
                       Examples: "openai/clip-vit-base-patch32",
                                "sentence-transformers/all-MiniLM-L6-v2"
            model_type: Type of model (clip or sentence_transformer)
            embedding_dimension: Dimension of vectors this model produces
                               Examples: 512 (CLIP), 384 (MiniLM), 768 (BERT)
        
        Returns:
            model_id: UUID of the registered model
        
        Example:
            # Register CLIP model for images
            model_id = await embedder.register_model(
                model_name="openai/clip-vit-base-patch32",
                model_type=ModelType.CLIP,
                embedding_dimension=512
            )
            
            # Register sentence transformer for text
            model_id = await embedder.register_model(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_type=ModelType.SENTENCE_TRANSFORMER,
                embedding_dimension=384
            )
        
        Notes:
            - If model already registered, returns existing model_id
            - Model name should be resolvable (e.g., Hugging Face model ID)
            - This doesn't necessarily load the model into memory; that may
              happen lazily on first use
        """
        pass
    
    @abstractmethod
    async def get_registered_models(
        self,
        model_type: Optional[ModelType] = None
    ) -> List[EmbeddingModel]:
        """
        Get all registered embedding models.
        
        Purpose:
            Lists all models available in the system, optionally filtered
            by type. Useful for UIs and validation.
        
        Args:
            model_type: Optional filter by model type
        
        Returns:
            List of EmbeddingModel dictionaries, each containing:
                - model_name: Name of the model
                - model_type: Type (clip or sentence_transformer)
                - embedding_dimension: Vector dimension
        
        Example:
            # Get all models
            all_models = await embedder.get_registered_models()
            
            # Get only CLIP models
            clip_models = await embedder.get_registered_models(
                model_type=ModelType.CLIP
            )
            
            for model in clip_models:
                print(f"{model['model_name']}: {model['embedding_dimension']}D")
        """
        pass
    
    @abstractmethod
    async def embed_image(
        self,
        image_path: str,
        model_name: Optional[str] = None
    ) -> List[float]:
        """
        Generate an embedding vector for an image.
        
        Purpose:
            Takes an image file path and produces an embedding vector using
            a CLIP or other vision model. This vector can be used for
            semantic image search.
        
        Args:
            image_path: Path to the image file
            model_name: Optional specific model to use. If None, uses the
                       default/primary image embedding model.
        
        Returns:
            List of floats representing the embedding vector. Length matches
            the model's embedding_dimension.
        
        Example:
            embedding = await embedder.embed_image(
                image_path="/data/photos/img001.jpg",
                model_name="openai/clip-vit-base-patch32"
            )
            
            print(f"Generated {len(embedding)}-dimensional embedding")
            # Output: Generated 512-dimensional embedding
        
        Raises:
            FileNotFoundError: If image_path doesn't exist
            ValueError: If model_name not registered or incompatible
            RuntimeError: If embedding generation fails
        
        Notes:
            - Image is automatically preprocessed (resized, normalized)
            - Supports common formats: JPEG, PNG, BMP, etc.
            - For CLIP models, this uses the vision encoder
        """
        pass
    
    @abstractmethod
    async def embed_text(
        self,
        text: str,
        model_name: Optional[str] = None
    ) -> List[float]:
        """
        Generate an embedding vector for text.
        
        Purpose:
            Takes a text string and produces an embedding vector using a
            sentence transformer or CLIP text encoder. This vector can be
            used for semantic text search or cross-modal search (text->image).
        
        Args:
            text: Text to embed (caption, description, query, document chunk)
            model_name: Optional specific model to use. If None, uses the
                       default/primary text embedding model.
        
        Returns:
            List of floats representing the embedding vector. Length matches
            the model's embedding_dimension.
        
        Example:
            # Embed a caption
            embedding = await embedder.embed_text(
                text="A person skiing down a snowy mountain",
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Embed for cross-modal search (text->image via CLIP)
            query_embedding = await embedder.embed_text(
                text="mountain resort",
                model_name="openai/clip-vit-base-patch32"
            )
        
        Raises:
            ValueError: If text is empty or model_name not registered
            RuntimeError: If embedding generation fails
        
        Notes:
            - Text is automatically tokenized and truncated to model max length
            - For sentence transformers, typically handles up to 512 tokens
            - For CLIP, this uses the text encoder
            - Long text may be truncated; consider chunking first
        """
        pass
    
    @abstractmethod
    async def embed_text_batch(
        self,
        texts: List[str],
        model_name: Optional[str] = None
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in a batch.
        
        Purpose:
            Efficiently processes multiple texts at once, which is much faster
            than calling embed_text repeatedly. Use this for chunked documents
            or batch processing.
        
        Args:
            texts: List of text strings to embed
            model_name: Optional specific model to use
        
        Returns:
            List of embedding vectors, one per input text, in the same order.
        
        Example:
            # Embed document chunks
            chunks = [
                "This is the first paragraph...",
                "This is the second paragraph...",
                "This is the third paragraph..."
            ]
            
            embeddings = await embedder.embed_text_batch(
                texts=chunks,
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            print(f"Generated {len(embeddings)} embeddings")
            # Output: Generated 3 embeddings
        
        Notes:
            - Much more efficient than sequential embed_text calls
            - Batching happens at the model level (GPU parallelism)
            - All texts must use the same model
        """
        pass
    
    @abstractmethod
    async def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Compute similarity between two embeddings.
        
        Purpose:
            Calculates cosine similarity between two embedding vectors.
            Useful for testing, validation, and offline analysis.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Cosine similarity score between 0 and 1:
                - 1.0: Identical or very similar
                - 0.5: Somewhat similar
                - 0.0: Orthogonal/unrelated
        
        Example:
            emb1 = await embedder.embed_text("skiing in mountains")
            emb2 = await embedder.embed_text("snowboarding in alps")
            
            similarity = await embedder.compute_similarity(emb1, emb2)
            print(f"Similarity: {similarity:.2f}")
            # Output: Similarity: 0.87
        
        Raises:
            ValueError: If embeddings have different dimensions
        
        Notes:
            - Uses cosine similarity (normalized dot product)
            - Embeddings must have the same dimension
            - This is the same metric used in semantic_search
        """
        pass
    
    @abstractmethod
    async def get_model_info(
        self,
        model_name: str
    ) -> Optional[EmbeddingModel]:
        """
        Get information about a specific model.
        
        Purpose:
            Retrieves metadata about a registered model.
        
        Args:
            model_name: Name of the model to look up
        
        Returns:
            EmbeddingModel dictionary or None if not found.
        
        Example:
            info = await embedder.get_model_info(
                model_name="openai/clip-vit-base-patch32"
            )
            
            if info:
                print(f"Type: {info['model_type']}")
                print(f"Dimension: {info['embedding_dimension']}")
        """
        pass
