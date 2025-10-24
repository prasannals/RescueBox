# Metadata Storage and Search Interfaces

This folder contains the complete set of interfaces required for the unified metadata storage and semantic search service.

## Overview

These interfaces define the contracts between different components of the system:
- **Plugins** write metadata via `MetadataWriter`
- **Search UI/API** queries via `SearchService` and reads via `MetadataReader`
- **Embedding generation** happens through `EmbeddingService`
- **Background indexing** is orchestrated by `IndexingService`

## Interface Files

### 1. `types.py`
Common types, enums, and TypedDicts used across all interfaces.

**Key Types:**
- `MediaType`: IMAGE, TEXT
- `ContentType`: full_text, chunk, caption, description
- `EmbeddingSource`: image, text
- `ModelType`: clip, sentence_transformer
- `FileAttribute`: Structure for exact-match attributes
- `TextContent`: Structure for text content
- `EmbeddingVector`: Structure for embedding vectors
- `SearchFilter`: Structure for search filters
- `SearchResult`: Structure for search results

### 2. `metadata_writer.py`
Interface for plugins to write metadata to storage.

**Primary Use Case:** Plugins call these methods to persist their analysis results.

**Key Methods:**
- `register_collection(name)` - Create a collection for files
- `register_file(collection_id, path, type)` - Register a file
- `write_attributes(file_id, attributes)` - Write exact-match attributes
- `write_text_content(file_id, content)` - Write text for embedding
- `write_embedding(file_id, embedding)` - Write embedding vector
- `update_file_indexed_timestamp(file_id)` - Mark file as indexed

**Example Usage:**
```python
# In a plugin
writer = MetadataWriter(...)

file_id = await writer.register_file(
    collection_id="xyz",
    file_path="/data/img.jpg",
    media_type=MediaType.IMAGE
)

await writer.write_attributes(
    file_id=file_id,
    attributes=[
        {"attribute_key": "gender", "attribute_value": "male", "plugin_source": "age_gender"},
        {"attribute_key": "age_min", "attribute_value": "25", "plugin_source": "age_gender"}
    ]
)
```

### 3. `metadata_reader.py`
Interface for reading metadata without performing searches.

**Primary Use Case:** Display file details, debug, inspect metadata.

**Key Methods:**
- `get_file_info(file_id)` - Get basic file metadata
- `get_file_attributes(file_id, plugin_source?)` - Get attributes for a file
- `get_attribute_value(file_id, key)` - Get specific attribute value
- `get_file_text_content(file_id, content_type?)` - Get text content
- `get_collection_files(collection_id, filters...)` - List files in collection
- `get_attribute_statistics(key, collection_id?)` - Get attribute distribution stats

**Example Usage:**
```python
reader = MetadataReader(...)

# Display file details
info = await reader.get_file_info(file_id="abc-123")
attributes = await reader.get_file_attributes(file_id="abc-123")

for attr in attributes:
    print(f"{attr['attribute_key']}: {attr['attribute_value']}")
```

### 4. `search_service.py`
Interface for searching files based on metadata and embeddings.

**Primary Use Case:** Core search functionality - exact match, semantic, and combined.

**Key Methods:**
- `exact_match_search(filters, operator)` - Search by exact attribute values
- `semantic_search(query_text, params...)` - Search by semantic similarity
- `combined_search(filters, query, strategy)` - Combine exact + semantic
- `search_by_file_similarity(file_id)` - Find similar files
- `count_results(filters, query)` - Count matches without retrieving

**Example Usage:**
```python
search = SearchService(...)

# Find males aged 25-32 in mountain scenes
results = await search.combined_search(
    attribute_filters=[
        {"attribute_key": "gender", "attribute_value": "male"},
        {"attribute_key": "age_min", "value_range": (25, 32)}
    ],
    semantic_query="mountain resort skiing",
    combine_strategy="FILTER_THEN_SEARCH",
    top_k=50,
    similarity_threshold=0.7
)

for result in results:
    print(f"{result['file_path']}: {result['similarity_score']:.2f}")
```

### 5. `embedding_service.py`
Interface for generating embeddings from images and text.

**Primary Use Case:** ML layer - embed images/text using CLIP, sentence transformers, etc.

**Key Methods:**
- `register_model(name, type, dimension)` - Register an embedding model
- `get_registered_models(type?)` - List available models
- `embed_image(path, model?)` - Generate image embedding
- `embed_text(text, model?)` - Generate text embedding
- `embed_text_batch(texts, model?)` - Batch embed multiple texts
- `compute_similarity(emb1, emb2)` - Calculate cosine similarity
- `get_model_info(name)` - Get model details

**Example Usage:**
```python
embedder = EmbeddingService(...)

# Register CLIP model
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

# Embed text for search
query_embedding = await embedder.embed_text(
    text="mountain resort",
    model_name="openai/clip-vit-base-patch32"
)
```

### 6. `indexing_service.py`
Interface for managing async indexing operations.

**Primary Use Case:** Orchestration layer - queue jobs, run plugins, track progress.

**Key Methods:**
- `index_file(file_id, plugins, force_reindex?)` - Index a single file
- `index_collection(collection_id, plugins, params...)` - Index all files in collection
- `generate_embeddings(file_id, sources, model?)` - Generate embeddings for a file
- `get_job_status(job_id)` - Check job progress
- `cancel_job(job_id)` - Cancel a running job
- `get_job_details(job_id)` - Get detailed execution log
- `list_jobs(status?, limit, offset)` - List jobs
- `retry_failed_job(job_id)` - Retry a failed job
- `get_plugin_statistics(plugin_name, days)` - Get plugin performance stats

**Example Usage:**
```python
indexer = IndexingService(...)

# Index a collection
job_id = await indexer.index_collection(
    collection_id="xyz-789",
    plugins=["age_and_gender_detection", "image_summary", "clip_embedder"],
    media_type=MediaType.IMAGE,
    batch_size=20
)

# Monitor progress
while True:
    status = await indexer.get_job_status(job_id)
    print(f"Progress: {status['progress_percentage']}%")
    
    if status['status'] in ['completed', 'failed']:
        break
    
    await asyncio.sleep(2)
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     INDEXING FLOW                            │
└─────────────────────────────────────────────────────────────┘

1. User triggers indexing via IndexingService
   ↓
2. IndexingService creates async job (Celery)
   ↓
3. Job runs plugins on file(s)
   ↓
4. Each plugin uses MetadataWriter to persist results:
   - write_attributes() for exact-match metadata
   - write_text_content() for text to be embedded
   ↓
5. EmbeddingService generates embeddings:
   - embed_image() for images
   - embed_text() for text content
   ↓
6. Embeddings written via MetadataWriter.write_embedding()
   ↓
7. File marked as indexed via update_file_indexed_timestamp()


┌─────────────────────────────────────────────────────────────┐
│                      SEARCH FLOW                             │
└─────────────────────────────────────────────────────────────┘

1. User submits search query via SearchService
   ↓
2. For semantic search:
   - EmbeddingService.embed_text() embeds the query
   ↓
3. SearchService queries database:
   - Exact match: filter on file_attribute table
   - Semantic: vector similarity on file_embedding table
   - Combined: both operations + intersection/union
   ↓
4. Results returned as SearchResult objects
   ↓
5. User can use MetadataReader to get full details:
   - get_file_attributes() for all attributes
   - get_file_text_content() for captions/descriptions
```

## Implementation Notes

### Database Implementation
These interfaces will be implemented against PostgreSQL with pgvector:
- `MetadataWriter` and `MetadataReader` use SQLAlchemy ORM
- `SearchService` uses raw SQL for complex queries + pgvector operators
- Vector similarity uses HNSW index for performance

### Async Implementation
All interfaces use async/await:
- Methods return coroutines
- Enables concurrent operations
- Non-blocking I/O for database access
- Compatible with FastAPI, asyncio, etc.

### Celery Integration
`IndexingService` uses Celery + RabbitMQ:
- Jobs queued to RabbitMQ
- Worker processes execute jobs
- Status tracked in database
- Supports distributed processing

### Error Handling
Implementations should:
- Raise specific exceptions (FileNotFoundError, ValueError, etc.)
- Log errors with context
- Support retries for transient failures
- Validate inputs before processing

## Testing Strategy

### Unit Tests
Each interface should have unit tests using mocks:
```python
# Test MetadataWriter
mock_db = Mock()
writer = MetadataWriterImpl(db=mock_db)

await writer.write_attributes(
    file_id="test",
    attributes=[{"attribute_key": "test", "attribute_value": "value", "plugin_source": "test"}]
)

mock_db.insert.assert_called_once()
```

### Integration Tests
Test full flows with real database:
```python
# Test full indexing flow
writer = MetadataWriter(db)
embedder = EmbeddingService()

# Register and index
file_id = await writer.register_file(...)
await writer.write_attributes(...)
embedding = await embedder.embed_image(...)
await writer.write_embedding(...)

# Verify via reader
attrs = await reader.get_file_attributes(file_id)
assert len(attrs) > 0
```

### End-to-End Tests
Test complete use cases:
```python
# Test: "find males aged 25-32 in mountain scenes"
# 1. Index test images with metadata
# 2. Perform combined search
# 3. Verify results match expectations
```

## Next Steps

1. **Implement Database Schema** (already designed)
2. **Implement MetadataWriter** (start here - needed by plugins)
3. **Implement MetadataReader** (needed for verification)
4. **Implement EmbeddingService** (needed for semantic search)
5. **Implement SearchService** (core functionality)
6. **Implement IndexingService** (orchestration)
7. **Update Plugins** to use MetadataWriter
8. **Create Search Plugin** using SearchService
9. **Build UI** for search and monitoring

## Questions?

- What plugins should attributes should be standardized?
- Should we version the schema for backwards compatibility?
- Performance targets for search (latency, throughput)?
- Authentication/authorization requirements?
