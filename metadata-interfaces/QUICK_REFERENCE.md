# Interface Quick Reference

Quick reference for all interfaces in the metadata storage system.

## MetadataWriter - For Plugins

**Purpose:** Write metadata to storage

```python
# Register collection and file
collection_id = await writer.register_collection("case_001")
file_id = await writer.register_file(collection_id, "/path/to/file.jpg", MediaType.IMAGE)

# Write attributes (exact-match searchable)
await writer.write_attributes(file_id, [
    {"attribute_key": "gender", "attribute_value": "male", "plugin_source": "my_plugin"},
    {"attribute_key": "age_min", "attribute_value": "25", "plugin_source": "my_plugin"}
])

# Write text content (for embedding)
text_id = await writer.write_text_content(file_id, {
    "text_content": "A person skiing",
    "content_type": ContentType.CAPTION,
    "chunk_index": None,
    "plugin_source": "my_plugin"
})

# Write embedding
await writer.write_embedding(file_id, {
    "embedding": [0.1, 0.2, ...],
    "model_name": "openai/clip-vit-base-patch32",
    "embedding_source": EmbeddingSource.IMAGE,
    "text_content_id": None,
    "plugin_source": "my_plugin"
})

# Mark as indexed
await writer.update_file_indexed_timestamp(file_id)
```

---

## MetadataReader - For Inspection

**Purpose:** Read metadata without searching

```python
# Get file info
info = await reader.get_file_info(file_id)

# Get all attributes
attrs = await reader.get_file_attributes(file_id)
attrs_from_plugin = await reader.get_file_attributes(file_id, plugin_source="age_gender")

# Get specific attribute
gender = await reader.get_attribute_value(file_id, "gender")

# Get text content
captions = await reader.get_file_text_content(file_id, content_type="caption")

# List files in collection
files = await reader.get_collection_files(collection_id, media_type=MediaType.IMAGE)

# Get attribute statistics
stats = await reader.get_attribute_statistics("gender", collection_id="xyz")
```

---

## SearchService - For Queries

**Purpose:** Search files by attributes and semantics

```python
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
    top_k=50,
    similarity_threshold=0.7
)

# Combined search (THE MAIN USE CASE)
results = await search.combined_search(
    attribute_filters=[
        {"attribute_key": "gender", "attribute_value": "male"},
        {"attribute_key": "age_min", "value_range": (25, 32)}
    ],
    semantic_query="mountain resort",
    combine_strategy="FILTER_THEN_SEARCH",
    top_k=50,
    similarity_threshold=0.7
)

# Find similar files
similar = await search.search_by_file_similarity(
    file_id="abc-123",
    embedding_source=EmbeddingSource.IMAGE,
    top_k=20
)

# Count results
count = await search.count_results(
    filters=[{"attribute_key": "gender", "attribute_value": "male"}]
)
```

---

## EmbeddingService - For ML

**Purpose:** Generate embeddings from images/text

```python
# Register model
await embedder.register_model(
    model_name="openai/clip-vit-base-patch32",
    model_type=ModelType.CLIP,
    embedding_dimension=512
)

# Embed image
embedding = await embedder.embed_image(
    image_path="/path/to/image.jpg",
    model_name="openai/clip-vit-base-patch32"
)

# Embed text
embedding = await embedder.embed_text(
    text="mountain resort skiing",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Batch embed texts (faster)
embeddings = await embedder.embed_text_batch(
    texts=["text1", "text2", "text3"],
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Compute similarity
similarity = await embedder.compute_similarity(embedding1, embedding2)

# List models
models = await embedder.get_registered_models(model_type=ModelType.CLIP)
```

---

## IndexingService - For Orchestration

**Purpose:** Manage async indexing jobs

```python
# Index single file
job_id = await indexer.index_file(
    file_id="abc-123",
    plugins=["age_gender", "image_summary"],
    force_reindex=False
)

# Index entire collection (MAIN METHOD)
job_id = await indexer.index_collection(
    collection_id="xyz-789",
    plugins=["age_gender", "clip_embedder"],
    media_type=MediaType.IMAGE,
    batch_size=20
)

# Generate embeddings separately
job_id = await indexer.generate_embeddings(
    file_id="abc-123",
    embedding_sources=["image", "text"]
)

# Monitor job
status = await indexer.get_job_status(job_id)
print(f"Progress: {status['progress_percentage']}%")

# Get detailed results
details = await indexer.get_job_details(job_id)

# Cancel job
await indexer.cancel_job(job_id)

# List jobs
active = await indexer.list_jobs(status="in_progress")

# Retry failed job
new_job = await indexer.retry_failed_job(failed_job_id)

# Get plugin stats
stats = await indexer.get_plugin_statistics("age_gender", days=30)
```

---

## Common Types

```python
# Enums
MediaType.IMAGE, MediaType.TEXT
ContentType.FULL_TEXT, ContentType.CHUNK, ContentType.CAPTION, ContentType.DESCRIPTION
EmbeddingSource.IMAGE, EmbeddingSource.TEXT
ModelType.CLIP, ModelType.SENTENCE_TRANSFORMER

# SearchFilter
{"attribute_key": "gender", "attribute_value": "male"}
{"attribute_key": "age", "value_in": ["25", "26", "27"]}
{"attribute_key": "age_min", "value_range": (25, 32)}

# SearchResult
{
    "file_id": "abc-123",
    "file_path": "/data/img.jpg",
    "media_type": MediaType.IMAGE,
    "collection_id": "xyz-789",
    "collection_name": "case_001",
    "similarity_score": 0.87,  # or None for exact match
    "matched_attributes": [...]
}
```

---

## Typical Plugin Flow

```python
class MyPlugin:
    def __init__(self, writer: MetadataWriter, embedder: EmbeddingService):
        self.writer = writer
        self.embedder = embedder
    
    async def process_file(self, file_id: str, file_path: str):
        # 1. Run analysis
        results = self.analyze(file_path)
        
        # 2. Write attributes
        await self.writer.write_attributes(file_id, results['attributes'])
        
        # 3. Write text if any
        if results['caption']:
            text_id = await self.writer.write_text_content(file_id, {
                "text_content": results['caption'],
                "content_type": ContentType.CAPTION,
                "chunk_index": None,
                "plugin_source": "my_plugin"
            })
        
        # 4. Generate embeddings
        embedding = await self.embedder.embed_image(file_path)
        await self.writer.write_embedding(file_id, {
            "embedding": embedding,
            "model_name": "openai/clip-vit-base-patch32",
            "embedding_source": EmbeddingSource.IMAGE,
            "text_content_id": None,
            "plugin_source": "my_plugin"
        })
        
        # 5. Mark complete
        await self.writer.update_file_indexed_timestamp(file_id)
```

---

## Typical Search Flow

```python
async def search_ui_handler(user_query: dict):
    # Parse user input
    filters = user_query.get('filters', [])
    semantic_query = user_query.get('query_text')
    
    # Execute search
    if filters and semantic_query:
        # Combined search
        results = await search.combined_search(
            attribute_filters=filters,
            semantic_query=semantic_query,
            combine_strategy="FILTER_THEN_SEARCH",
            top_k=50,
            similarity_threshold=0.7
        )
    elif filters:
        # Just filters
        results = await search.exact_match_search(filters=filters, operator="AND")
    else:
        # Just semantic
        results = await search.semantic_search(
            query_text=semantic_query,
            top_k=50,
            similarity_threshold=0.7
        )
    
    # Enrich results
    for result in results:
        # Get full attributes for display
        result['all_attributes'] = await reader.get_file_attributes(result['file_id'])
        
        # Get captions
        captions = await reader.get_file_text_content(
            result['file_id'],
            content_type="caption"
        )
        if captions:
            result['caption'] = captions[0]['text_content']
    
    return results
```
