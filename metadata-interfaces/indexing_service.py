"""
IndexingService Interface

This interface manages the async indexing process. It handles:
- Queuing indexing jobs for files and collections
- Running plugins on files
- Coordinating between plugins, embedding generation, and storage
- Tracking job status and progress

This is the orchestration layer that ties everything together.

Usage Example:
    indexer = IndexingService(...)
    
    # Index a single file
    job_id = await indexer.index_file(
        file_id="abc-123",
        plugins=["age_and_gender_detection", "image_summary"]
    )
    
    # Index an entire collection
    job_id = await indexer.index_collection(
        collection_id="xyz-789",
        plugins=["age_and_gender_detection", "clip_embedder"]
    )
    
    # Check job status
    status = await indexer.get_job_status(job_id)
    print(f"Progress: {status['progress_percentage']}%")
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from .types import MediaType


class IndexingService(ABC):
    """
    Interface for managing async indexing operations.
    
    This service orchestrates the indexing pipeline: running plugins,
    generating embeddings, and storing metadata. It uses an async task
    queue (Celery + RabbitMQ) for background processing.
    """
    
    @abstractmethod
    async def index_file(
        self,
        file_id: str,
        plugins: List[str],
        force_reindex: bool = False
    ) -> str:
        """
        Queue an indexing job for a single file.
        
        Purpose:
            Creates an async job to run specified plugins on a file. The job
            executes in the background, allowing the caller to continue without
            blocking. Plugins run sequentially in the order specified.
        
        Args:
            file_id: UUID of the file to index
            plugins: List of plugin names to run (e.g., ["age_and_gender_detection"])
            force_reindex: If True, reindex even if file was already indexed.
                          If False, skips if file.indexed_at is recent.
        
        Returns:
            job_id: UUID of the created indexing job
        
        Example:
            # Index a single image
            job_id = await indexer.index_file(
                file_id="abc-123",
                plugins=[
                    "age_and_gender_detection",
                    "image_summary",
                    "clip_embedder"
                ]
            )
            
            print(f"Job queued: {job_id}")
            
            # Force reindexing
            job_id = await indexer.index_file(
                file_id="abc-123",
                plugins=["age_and_gender_detection"],
                force_reindex=True
            )
        
        Notes:
            - Job executes asynchronously; use get_job_status to monitor
            - If file doesn't exist, job fails immediately
            - Invalid plugin names cause job to fail
            - Each plugin writes metadata via MetadataWriter interface
        """
        pass
    
    @abstractmethod
    async def index_collection(
        self,
        collection_id: str,
        plugins: List[str],
        media_type: Optional[MediaType] = None,
        force_reindex: bool = False,
        batch_size: int = 10
    ) -> str:
        """
        Queue an indexing job for all files in a collection.
        
        Purpose:
            Creates an async job to index multiple files in a collection.
            Files are processed in batches for efficiency. This is the primary
            way to bulk-index evidence folders.
        
        Args:
            collection_id: UUID of the collection to index
            plugins: List of plugin names to run on each file
            media_type: Optional filter - only index files of this type
            force_reindex: If True, reindex all files. If False, skip already
                          indexed files.
            batch_size: Number of files to process in parallel per batch
        
        Returns:
            job_id: UUID of the created indexing job
        
        Example:
            # Index all images in a collection
            job_id = await indexer.index_collection(
                collection_id="xyz-789",
                plugins=[
                    "age_and_gender_detection",
                    "clip_embedder"
                ],
                media_type=MediaType.IMAGE,
                batch_size=20
            )
            
            # Index all files (images and text)
            job_id = await indexer.index_collection(
                collection_id="xyz-789",
                plugins=["doc_parser", "text_embedder", "clip_embedder"]
            )
        
        Notes:
            - Job processes files in batches for memory efficiency
            - Progress updates as each batch completes
            - If one file fails, job continues with remaining files
            - Final job status includes count of successes/failures
        """
        pass
    
    @abstractmethod
    async def generate_embeddings(
        self,
        file_id: str,
        embedding_sources: List[str],
        model_name: Optional[str] = None
    ) -> str:
        """
        Queue a job to generate embeddings for a file.
        
        Purpose:
            Generates embeddings from existing file content (image, text content).
            This can be run separately from plugin indexing, useful for:
            - Regenerating embeddings with a new model
            - Creating embeddings after text content was added by plugins
        
        Args:
            file_id: UUID of the file
            embedding_sources: Which sources to embed - "image", "text", or both
            model_name: Optional specific model to use. If None, uses defaults.
        
        Returns:
            job_id: UUID of the created job
        
        Example:
            # Generate image embedding
            job_id = await indexer.generate_embeddings(
                file_id="abc-123",
                embedding_sources=["image"],
                model_name="openai/clip-vit-base-patch32"
            )
            
            # Generate embeddings for all text content
            job_id = await indexer.generate_embeddings(
                file_id="abc-123",
                embedding_sources=["text"],
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Generate both
            job_id = await indexer.generate_embeddings(
                file_id="abc-123",
                embedding_sources=["image", "text"]
            )
        
        Notes:
            - For text embeddings, embeds all text_content entries for the file
            - Existing embeddings for the same model are replaced
            - If file has no text content, text embedding is skipped
        """
        pass
    
    @abstractmethod
    async def get_job_status(
        self,
        job_id: str
    ) -> Dict[str, Any]:
        """
        Get the current status of an indexing job.
        
        Purpose:
            Retrieves detailed information about a job's progress and status.
            Used for monitoring long-running indexing operations.
        
        Args:
            job_id: UUID of the job
        
        Returns:
            Dictionary containing:
                - job_id: UUID of the job
                - job_type: Type of job (index_file, index_collection, etc.)
                - status: Current status (queued, in_progress, completed, failed)
                - progress_percentage: Progress from 0-100
                - files_processed: Number of files completed
                - files_total: Total number of files to process
                - started_at: When job started (or None)
                - completed_at: When job finished (or None)
                - error_message: Error details if failed (or None)
            Returns None if job doesn't exist.
        
        Example:
            status = await indexer.get_job_status(job_id="job-456")
            
            if status:
                print(f"Status: {status['status']}")
                print(f"Progress: {status['progress_percentage']}%")
                
                if status['status'] == 'completed':
                    print("Job finished successfully!")
                elif status['status'] == 'failed':
                    print(f"Job failed: {status['error_message']}")
        
        Notes:
            - Progress updates in real-time
            - For collection jobs, progress = (files_processed / files_total) * 100
            - completed status doesn't mean all files succeeded; check details
        """
        pass
    
    @abstractmethod
    async def cancel_job(
        self,
        job_id: str
    ) -> bool:
        """
        Cancel a running or queued job.
        
        Purpose:
            Attempts to stop a job. If queued, removes from queue. If running,
            signals cancellation (current file may finish).
        
        Args:
            job_id: UUID of the job to cancel
        
        Returns:
            True if job was cancelled, False if job doesn't exist or already done
        
        Example:
            success = await indexer.cancel_job(job_id="job-456")
            
            if success:
                print("Job cancelled")
            else:
                print("Job could not be cancelled (already done?)")
        
        Notes:
            - Queued jobs cancel immediately
            - Running jobs may take a moment to stop
            - Completed/failed jobs cannot be cancelled
        """
        pass
    
    @abstractmethod
    async def get_job_details(
        self,
        job_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed execution log for a job.
        
        Purpose:
            Retrieves detailed per-file execution results, including which
            plugins ran, how long they took, and any errors. Useful for
            debugging and auditing.
        
        Args:
            job_id: UUID of the job
        
        Returns:
            Dictionary containing:
                - job_id: UUID of the job
                - job_status: Overall job status
                - file_executions: List of per-file results, each containing:
                    - file_id: UUID of the file
                    - file_path: Path to the file
                    - plugin_results: List of plugin executions:
                        - plugin_name: Name of the plugin
                        - status: success, failed, or skipped
                        - execution_time_ms: Time taken
                        - error_message: Error if failed
                        - attributes_created: Count of attributes written
                        - embeddings_created: Count of embeddings written
            Returns None if job doesn't exist.
        
        Example:
            details = await indexer.get_job_details(job_id="job-456")
            
            if details:
                for file_exec in details['file_executions']:
                    print(f"File: {file_exec['file_path']}")
                    
                    for plugin in file_exec['plugin_results']:
                        if plugin['status'] == 'failed':
                            print(f"  {plugin['plugin_name']} FAILED: {plugin['error_message']}")
                        else:
                            print(f"  {plugin['plugin_name']}: {plugin['execution_time_ms']}ms")
        
        Notes:
            - Only available after job starts executing
            - Useful for identifying problematic files or plugins
            - Can be large for collection jobs; may be paginated in future
        """
        pass
    
    @abstractmethod
    async def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List indexing jobs with optional filtering.
        
        Purpose:
            Retrieves a list of jobs for monitoring, history, or cleanup.
        
        Args:
            status: Optional filter by status (queued, in_progress, completed, failed)
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip (for pagination)
        
        Returns:
            List of job summary dictionaries (same format as get_job_status).
            Ordered by created_at (most recent first).
        
        Example:
            # Get all in-progress jobs
            active_jobs = await indexer.list_jobs(status="in_progress")
            
            for job in active_jobs:
                print(f"{job['job_id']}: {job['progress_percentage']}%")
            
            # Get recent completed jobs
            recent = await indexer.list_jobs(status="completed", limit=10)
        
        Notes:
            - Use for building job management UIs
            - Pagination recommended for large result sets
        """
        pass
    
    @abstractmethod
    async def retry_failed_job(
        self,
        job_id: str
    ) -> str:
        """
        Retry a failed job.
        
        Purpose:
            Creates a new job that retries only the failed files from a
            previous job. Useful for recovering from transient errors.
        
        Args:
            job_id: UUID of the failed job to retry
        
        Returns:
            new_job_id: UUID of the new retry job
        
        Example:
            # Original job failed
            status = await indexer.get_job_status("job-456")
            if status['status'] == 'failed':
                # Retry it
                new_job_id = await indexer.retry_failed_job("job-456")
                print(f"Retry job: {new_job_id}")
        
        Raises:
            ValueError: If original job is not in failed status
        
        Notes:
            - Only retries files that failed, skips successful ones
            - Uses same plugins and settings as original job
            - Original job remains in history unchanged
        """
        pass
    
    @abstractmethod
    async def get_plugin_statistics(
        self,
        plugin_name: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get statistics about a plugin's performance.
        
        Purpose:
            Provides insights into plugin execution patterns, useful for
            monitoring and optimization.
        
        Args:
            plugin_name: Name of the plugin to analyze
            days: Number of days of history to include
        
        Returns:
            Dictionary containing:
                - plugin_name: Name of the plugin
                - total_executions: Total number of times run
                - success_count: Number of successful runs
                - failure_count: Number of failed runs
                - average_execution_time_ms: Average time per execution
                - median_execution_time_ms: Median time per execution
                - total_attributes_created: Total attributes written
                - total_embeddings_created: Total embeddings written
        
        Example:
            stats = await indexer.get_plugin_statistics(
                plugin_name="age_and_gender_detection",
                days=7
            )
            
            print(f"Success rate: {stats['success_count'] / stats['total_executions']:.1%}")
            print(f"Average time: {stats['average_execution_time_ms']}ms")
        
        Notes:
            - Useful for identifying slow or unreliable plugins
            - Can guide optimization efforts
        """
        pass
