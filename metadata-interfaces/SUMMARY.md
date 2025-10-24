# Metadata Storage and Search Interfaces - Summary

## What Was Created

A complete set of Python interface definitions for the unified metadata storage and semantic search service. These interfaces define the contracts between all components of the system.

## Files Created

1. **`__init__.py`** - Package initialization with exports
2. **`types.py`** - Common types, enums, and TypedDicts
3. **`metadata_writer.py`** - Interface for plugins to write metadata
4. **`metadata_reader.py`** - Interface for reading metadata
5. **`search_service.py`** - Interface for searching files
6. **`embedding_service.py`** - Interface for generating embeddings
7. **`indexing_service.py`** - Interface for async job orchestration
8. **`README.md`** - Comprehensive documentation
9. **`QUICK_REFERENCE.md`** - Quick reference guide
10. **`plugin_integration_example.py`** - Complete examples

## Core Interfaces

### 1. MetadataWriter
**For:** Plugins that extract metadata
**Purpose:** Write attributes, text content, and embeddings to storage

**Key Methods:**
- `register_collection(name)` - Create collection
- `register_file(collection_id, path, type)` - Register file
- `write_attributes(file_id, attributes)` - Write searchable attributes
- `write_text_content(file_id, content)` - Write text for embedding
- `write_embedding(file_id, embedding)` - Write embedding vector
- `update_file_indexed_timestamp(file_id)` - Mark as indexed

### 2. MetadataReader
**For:** UIs, debugging, inspection
**Purpose:** Read metadata without performing searches

**Key Methods:**
- `get_file_info(file_id)` - Get file metadata
- `get_file_attributes(file_id, plugin_source?)` - Get attributes
- `get_attribute_value(file_id, key)` - Get specific attribute
- `get_file_text_content(file_id, type?)` - Get text content
- `get_collection_files(collection_id, filters...)` - List files
- `get_attribute_statistics(key, collection_id?)` - Get stats

### 3. SearchService
**For:** Search UIs, APIs
**Purpose:** Execute searches using attributes and semantics

**Key Methods:**
- `exact_match_search(filters, operator)` - Search by attributes
- `semantic_search(query_text, params...)` - Search by similarity
- `combined_search(filters, query, strategy)` - Combined search (MAIN)
- `search_by_file_similarity(file_id)` - Find similar files
- `count_results(filters, query)` - Count matches

### 4. EmbeddingService
**For:** ML operations
**Purpose:** Generate embeddings from images and text

**Key Methods:**
- `register_model(name, type, dimension)` - Register model
- `get_registered_models(type?)` - List models
- `embed_image(path, model?)` - Embed image
- `embed_text(text, model?)` - Embed text
- `embed_text_batch(texts, model?)` - Batch embed
- `compute_similarity(emb1, emb2)` - Calculate similarity

### 5. IndexingService
**For:** Job orchestration
**Purpose:** Manage async indexing operations

**Key Methods:**
- `index_file(file_id, plugins, force?)` - Index single file
- `index_collection(collection_id, plugins, params...)` - Index collection
- `generate_embeddings(file_id, sources, model?)` - Generate embeddings
- `get_job_status(job_id)` - Check progress
- `get_job_details(job_id)` - Get execution log
- `list_jobs(status?, limit, offset)` - List jobs
- `retry_failed_job(job_id)` - Retry failed job
- `get_plugin_statistics(plugin, days)` - Get stats

## Key Design Principles

1. **Separation of Concerns**
   - Writing metadata (MetadataWriter) is separate from reading (MetadataReader)
   - Searching (SearchService) is separate from indexing (IndexingService)
   - ML operations (EmbeddingService) are isolated

2. **Async/Await**
   - All methods are async for non-blocking I/O
   - Enables concurrent operations
   - Compatible with modern Python frameworks

3. **Type Safety**
   - TypedDicts for structured data
   - Enums for valid values
   - Clear parameter types and return types

4. **Plugin-Friendly**
   - Simple interface for plugins to use
   - MetadataWriter handles all storage complexity
   - Plugins don't need database knowledge

5. **Flexible Search**
   - Exact match for filtering
   - Semantic search for relevance
   - Combined search for complex queries
   - Support for the core use case: "find males aged 25-32 in mountain resort areas"

## Data Flow

```
INDEXING FLOW:
User → IndexingService → Celery Job → Plugin → MetadataWriter → Database
                                    ↓
                              EmbeddingService → MetadataWriter → Database

SEARCH FLOW:
User → SearchService → EmbeddingService (for query) → Database → Results
                    ↓
              MetadataReader (for details) → Results
```

## Example Use Case: "Find males aged 25-32 in mountain resort areas"

```python
# Combined search using SearchService
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
```

This single call:
1. Filters files where gender="male" AND age is 25-32
2. Among filtered files, searches semantically for "mountain resort skiing snow"
3. Returns top 50 results with similarity > 0.7
4. Ordered by semantic similarity

## Next Steps for Implementation

### Phase 1: Database & Core Storage
1. Implement PostgreSQL schema (already designed)
2. Set up pgvector extension
3. Implement MetadataWriter (synchronous version first)
4. Implement MetadataReader
5. Write tests for write/read operations

### Phase 2: Embeddings
6. Implement EmbeddingService
   - CLIP for images
   - Sentence transformers for text
7. Test embedding generation
8. Test vector storage in pgvector

### Phase 3: Search
9. Implement SearchService
   - Start with exact_match_search
   - Add semantic_search
   - Add combined_search
10. Test search queries
11. Optimize with indexes

### Phase 4: Async Orchestration
12. Set up Celery + RabbitMQ
13. Implement IndexingService
14. Create Celery tasks
15. Test async indexing

### Phase 5: Plugin Integration
16. Update existing plugins to use MetadataWriter
17. Test plugin integration
18. Create new plugins as needed

### Phase 6: Search UI/Plugin
19. Create search plugin using SearchService
20. Build UI for search
21. Add monitoring for IndexingService jobs

## Technology Stack

- **Database:** PostgreSQL 15+
- **Vector Storage:** pgvector extension
- **ORM:** SQLAlchemy (async)
- **Task Queue:** Celery + RabbitMQ
- **Image Embeddings:** OpenCLIP (or similar)
- **Text Embeddings:** sentence-transformers
- **Web Framework:** FastAPI (for API layer)

## Success Criteria

✅ Plugins can write metadata with simple API
✅ Exact-match filtering works efficiently
✅ Semantic search returns relevant results
✅ Combined queries work as expected
✅ Async indexing scales to large collections
✅ Search latency < 500ms for typical queries
✅ System handles thousands of files per collection

## Questions to Consider

1. **Attribute Schema:** Should we pre-define a schema for common attributes (gender, age, etc.) or keep it fully dynamic?
2. **Performance:** What are the target performance metrics? (files/sec for indexing, ms for search)
3. **Versioning:** How do we handle model upgrades? Re-embed everything?
4. **Authorization:** Do search results need to respect file permissions?
5. **Monitoring:** What metrics should we expose? (indexing rate, search latency, etc.)

## Contact & Feedback

These interfaces are designed to be:
- **Clear:** Well-documented with examples
- **Flexible:** Support current needs without over-engineering
- **Testable:** Easy to mock and unit test
- **Extensible:** Can add features without breaking existing code

Review the interfaces and provide feedback on:
- Missing functionality
- Confusing APIs
- Parameter naming
- Documentation clarity
