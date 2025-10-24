# Metadata Interfaces - File Index

Complete list of all files in this folder with descriptions.

## Core Interface Files

### `__init__.py`
Package initialization file that exports all interfaces.

### `types.py` (148 lines)
Common types, enums, and TypedDicts used across all interfaces.
- Enums: `MediaType`, `ContentType`, `EmbeddingSource`, `ModelType`
- TypedDicts: `FileAttribute`, `TextContent`, `EmbeddingVector`, `SearchFilter`, `SearchResult`, `EmbeddingModel`

### `metadata_writer.py` (229 lines)
**Interface for plugins to write metadata to storage.**

6 methods:
1. `register_collection(name)` - Create a collection
2. `register_file(collection_id, path, type)` - Register a file
3. `write_attributes(file_id, attributes)` - Write exact-match attributes
4. `write_text_content(file_id, content)` - Write text for embedding
5. `write_embedding(file_id, embedding)` - Write embedding vector
6. `update_file_indexed_timestamp(file_id)` - Mark as indexed

### `metadata_reader.py` (216 lines)
**Interface for reading metadata without performing searches.**

8 methods:
1. `get_file_info(file_id)` - Get basic file metadata
2. `get_file_attributes(file_id, plugin_source?)` - Get attributes
3. `get_attribute_value(file_id, key)` - Get specific attribute value
4. `get_file_text_content(file_id, content_type?)` - Get text content
5. `get_collection_files(collection_id, filters...)` - List files in collection
6. `get_files_by_path_pattern(pattern)` - Find files by path
7. `get_attribute_statistics(key, collection_id?)` - Get attribute stats
8. Additional read operations for inspection and debugging

### `search_service.py` (309 lines)
**Interface for searching files based on metadata and embeddings.**

5 methods:
1. `exact_match_search(filters, operator, ...)` - Search by attributes
2. `semantic_search(query_text, params...)` - Search by similarity
3. `combined_search(filters, query, strategy, ...)` - Combined search ⭐ MAIN METHOD
4. `search_by_file_similarity(file_id, ...)` - Find similar files
5. `count_results(filters, query, ...)` - Count matches

### `embedding_service.py` (258 lines)
**Interface for generating embeddings from images and text.**

8 methods:
1. `register_model(name, type, dimension)` - Register embedding model
2. `get_registered_models(type?)` - List available models
3. `embed_image(path, model?)` - Generate image embedding
4. `embed_text(text, model?)` - Generate text embedding
5. `embed_text_batch(texts, model?)` - Batch embed texts (efficient)
6. `compute_similarity(emb1, emb2)` - Calculate cosine similarity
7. `get_model_info(name)` - Get model details
8. Model management operations

### `indexing_service.py` (354 lines)
**Interface for managing async indexing operations.**

10 methods:
1. `index_file(file_id, plugins, force?)` - Index single file
2. `index_collection(collection_id, plugins, ...)` - Index collection ⭐ MAIN METHOD
3. `generate_embeddings(file_id, sources, model?)` - Generate embeddings
4. `get_job_status(job_id)` - Check job progress
5. `cancel_job(job_id)` - Cancel running job
6. `get_job_details(job_id)` - Get detailed execution log
7. `list_jobs(status?, limit, offset)` - List jobs
8. `retry_failed_job(job_id)` - Retry failed job
9. `get_plugin_statistics(plugin, days)` - Get plugin performance stats
10. Job orchestration and monitoring

## Documentation Files

### `README.md` (259 lines)
**Comprehensive documentation of the entire interface system.**

Covers:
- Overview of all interfaces
- Purpose and key methods for each interface
- Usage examples
- Data flow diagrams
- Implementation notes
- Testing strategy
- Next steps

### `SUMMARY.md` (220 lines)
**High-level summary of the interface design.**

Includes:
- What was created
- Core interface descriptions
- Key design principles
- Example use case walkthrough
- Implementation phases
- Technology stack
- Success criteria

### `QUICK_REFERENCE.md` (180 lines)
**Quick reference guide for developers.**

Contains:
- Code snippets for each interface
- Common usage patterns
- Typical plugin flow
- Typical search flow
- Common types reference

### `ARCHITECTURE.md` (380 lines)
**Visual diagrams and architecture documentation.**

Includes:
- Component overview diagram
- Interface layer diagram
- Data flow diagrams (indexing and search)
- Plugin integration pattern
- Search query execution plans
- Deployment architecture

### `plugin_integration_example.py` (362 lines)
**Complete working examples of plugin integration.**

Contains:
- `ExamplePlugin` class - Shows how to integrate with metadata storage
- `TextDocumentPlugin` class - Shows text document processing
- `example_indexing_flow()` - Complete indexing workflow
- `example_search_flow()` - Complete search workflow

Demonstrates:
- How plugins use MetadataWriter
- How to handle images and text
- How to generate embeddings
- How to perform searches
- Real-world usage patterns

## File Statistics

- **Total Files:** 12
- **Interface Files:** 6 (types + 5 interfaces)
- **Documentation Files:** 4 (README, SUMMARY, QUICK_REFERENCE, ARCHITECTURE)
- **Example Files:** 1 (plugin_integration_example)
- **Package Files:** 1 (__init__.py)

## Lines of Code

- Interface definitions: ~1,514 lines
- Documentation: ~1,039 lines
- Examples: ~362 lines
- **Total: ~2,915 lines**

## How to Use This Folder

1. **Start with:** `SUMMARY.md` - Get the big picture
2. **Then read:** `README.md` - Understand each interface
3. **For quick lookup:** `QUICK_REFERENCE.md` - Code snippets
4. **For diagrams:** `ARCHITECTURE.md` - Visual understanding
5. **For examples:** `plugin_integration_example.py` - See it in action

## Interface Dependencies

```
Plugin → MetadataWriter → Database
         MetadataWriter → EmbeddingService → Database

SearchUI → SearchService → Database
           SearchService → EmbeddingService
           MetadataReader → Database

Admin → IndexingService → Celery → Plugin → MetadataWriter
```

## Key Interfaces for Different Roles

**Plugin Developer:**
- Primary: `metadata_writer.py`, `types.py`
- Secondary: `embedding_service.py`
- Reference: `plugin_integration_example.py`

**Search UI Developer:**
- Primary: `search_service.py`, `metadata_reader.py`
- Secondary: `types.py`
- Reference: `QUICK_REFERENCE.md`

**System Administrator:**
- Primary: `indexing_service.py`
- Secondary: All interfaces
- Reference: `ARCHITECTURE.md`

**ML Engineer:**
- Primary: `embedding_service.py`
- Secondary: `types.py`
- Reference: `README.md`

## Implementation Checklist

- [ ] Implement `types.py` enums and TypedDicts
- [ ] Implement `MetadataWriter` with PostgreSQL backend
- [ ] Implement `MetadataReader` with PostgreSQL backend
- [ ] Implement `EmbeddingService` with CLIP and SentenceTransformers
- [ ] Implement `SearchService` with pgvector
- [ ] Implement `IndexingService` with Celery
- [ ] Write unit tests for each interface
- [ ] Write integration tests
- [ ] Update existing plugins to use new interfaces
- [ ] Create search plugin
- [ ] Build monitoring UI for IndexingService
- [ ] Performance testing and optimization

## Contact

For questions about:
- **Interface design:** Review `README.md` and `ARCHITECTURE.md`
- **Usage examples:** See `plugin_integration_example.py` and `QUICK_REFERENCE.md`
- **Implementation:** Follow the phases in `SUMMARY.md`
