# System Architecture Diagram

## Component Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACES                              │
│  - Search UI                                                         │
│  - Indexing UI                                                       │
│  - File Browser                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌──────────────────────┐       ┌──────────────────────┐
        │   SearchService      │       │  IndexingService     │
        │                      │       │                      │
        │  - exact_match       │       │  - index_file        │
        │  - semantic_search   │       │  - index_collection  │
        │  - combined_search   │       │  - get_job_status    │
        └──────────────────────┘       └──────────────────────┘
                    │                               │
                    │                               ▼
                    │                   ┌──────────────────────┐
                    │                   │   Celery + RabbitMQ  │
                    │                   │   (Task Queue)       │
                    │                   └──────────────────────┘
                    │                               │
                    │                               ▼
                    │                   ┌──────────────────────┐
                    │                   │   Worker Processes   │
                    │                   │   (Run Plugins)      │
                    │                   └──────────────────────┘
                    │                               │
                    │                               ▼
                    │                   ┌──────────────────────┐
                    │                   │   PLUGINS            │
                    │                   │  - age_gender        │
                    │                   │  - image_summary     │
                    │                   │  - face_recognition  │
                    │                   │  - doc_parser        │
                    │                   └──────────────────────┘
                    │                               │
                    │                               ▼
                    │                   ┌──────────────────────┐
                    │                   │  MetadataWriter      │
                    │                   │                      │
                    │                   │  - write_attributes  │
                    │                   │  - write_text_content│
                    │                   │  - write_embedding   │
                    │                   └──────────────────────┘
                    │                               │
                    │                               │
                    ▼                               ▼
        ┌──────────────────────────────────────────────────┐
        │            EmbeddingService                       │
        │                                                   │
        │  - embed_image()                                  │
        │  - embed_text()                                   │
        │  - Models: CLIP, SentenceTransformers            │
        └──────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌──────────────────────────────────────────────────┐
        │       PostgreSQL + pgvector                       │
        │                                                   │
        │  Tables:                                          │
        │  - media_collection                               │
        │  - media_file                                     │
        │  - file_attribute (exact match)                   │
        │  - file_text_content                              │
        │  - file_embedding (HNSW index)                    │
        │  - embedding_model                                │
        └──────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌──────────────────────────────────────────────────┐
        │          MetadataReader                           │
        │                                                   │
        │  - get_file_info()                                │
        │  - get_file_attributes()                          │
        │  - get_file_text_content()                        │
        └──────────────────────────────────────────────────┘
```

## Interface Layer Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      INTERFACE LAYER                             │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Metadata  │  │   Metadata  │  │   Search    │             │
│  │   Writer    │  │   Reader    │  │   Service   │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐                               │
│  │  Embedding  │  │   Indexing  │                               │
│  │   Service   │  │   Service   │                               │
│  └─────────────┘  └─────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ All interfaces are abstract
                              │ Implementations inject dependencies
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   IMPLEMENTATION LAYER                           │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  SQLAlchemy Models + Database Operations             │       │
│  │  - Create/Read/Update operations                     │       │
│  │  - Transaction management                            │       │
│  │  - Connection pooling                                │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  ML Model Implementations                            │       │
│  │  - CLIP loading and inference                        │       │
│  │  - SentenceTransformer loading and inference         │       │
│  │  - Model caching and batching                        │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  Celery Tasks                                        │       │
│  │  - Task definitions                                  │       │
│  │  - Progress tracking                                 │       │
│  │  - Error handling and retries                        │       │
│  └──────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow: Indexing a File

```
┌──────┐
│ User │ "Index this folder"
└──┬───┘
   │
   ▼
┌────────────────────────┐
│ IndexingService        │
│ .index_collection()    │
└────────────────────────┘
   │
   │ Creates Celery job
   │
   ▼
┌────────────────────────┐
│ Celery Worker          │
│ Picks up job           │
└────────────────────────┘
   │
   │ For each file:
   │
   ▼
┌────────────────────────┐
│ Plugin                 │
│ (e.g., age_gender)     │
└────────────────────────┘
   │
   │ 1. Analyze file
   │ 2. Extract metadata
   │
   ▼
┌────────────────────────┐
│ MetadataWriter         │
│ .write_attributes()    │
└────────────────────────┘
   │
   ▼
┌────────────────────────┐
│ Database               │
│ file_attribute table   │
└────────────────────────┘
   │
   ▼
┌────────────────────────┐
│ Plugin                 │
│ Generate caption       │
└────────────────────────┘
   │
   ▼
┌────────────────────────┐
│ MetadataWriter         │
│ .write_text_content()  │
└────────────────────────┘
   │
   ▼
┌────────────────────────┐
│ Database               │
│ file_text_content      │
└────────────────────────┘
   │
   ▼
┌────────────────────────┐
│ EmbeddingService       │
│ .embed_image()         │
│ .embed_text()          │
└────────────────────────┘
   │
   ▼
┌────────────────────────┐
│ MetadataWriter         │
│ .write_embedding()     │
└────────────────────────┘
   │
   ▼
┌────────────────────────┐
│ Database               │
│ file_embedding table   │
└────────────────────────┘
```

## Data Flow: Searching Files

```
┌──────┐
│ User │ "Find males aged 25-32 in mountains"
└──┬───┘
   │
   ▼
┌────────────────────────────────────────┐
│ SearchService                          │
│ .combined_search(                      │
│   filters=[gender, age_min, age_max],  │
│   query="mountain resort"              │
│ )                                      │
└────────────────────────────────────────┘
   │
   │ Step 1: Exact match filtering
   │
   ▼
┌────────────────────────┐
│ Database               │
│ SELECT from            │
│ file_attribute         │
│ WHERE ...              │
└────────────────────────┘
   │
   │ Result: file_ids matching filters
   │
   ▼
┌────────────────────────┐
│ SearchService          │
│ Filter results         │
└────────────────────────┘
   │
   │ Step 2: Embed query
   │
   ▼
┌────────────────────────┐
│ EmbeddingService       │
│ .embed_text(           │
│   "mountain resort"    │
│ )                      │
└────────────────────────┘
   │
   │ Result: query_embedding
   │
   ▼
┌────────────────────────┐
│ Database               │
│ SELECT from            │
│ file_embedding         │
│ WHERE file_id IN (...)  │
│ ORDER BY embedding     │
│ <=> query_embedding    │
└────────────────────────┘
   │
   │ Result: ranked file_ids
   │
   ▼
┌────────────────────────┐
│ SearchService          │
│ Build SearchResults    │
└────────────────────────┘
   │
   ▼
┌────────────────────────┐
│ User gets results      │
│ with similarity scores │
└────────────────────────┘
```

## Plugin Integration Pattern

```
┌─────────────────────────────────────────────────────────┐
│                    PLUGIN                                │
│                                                          │
│  class MyPlugin:                                         │
│      def __init__(self, writer, embedder):               │
│          self.writer = writer                            │
│          self.embedder = embedder                        │
│                                                          │
│      async def process(self, file_id, file_path):        │
│          # 1. Run ML model                               │
│          results = await self.analyze(file_path)         │
│                                                          │
│          # 2. Write to storage via interface             │
│          await self.writer.write_attributes(...)         │
│          await self.writer.write_text_content(...)       │
│                                                          │
│          # 3. Generate embeddings via interface          │
│          emb = await self.embedder.embed_image(...)      │
│          await self.writer.write_embedding(...)          │
│                                                          │
│          # 4. Mark complete                              │
│          await self.writer.update_file_indexed_timestamp │
└─────────────────────────────────────────────────────────┘
                              │
                              │ Uses interfaces only
                              │ No direct database access
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│               INTERFACE IMPLEMENTATIONS                  │
│                                                          │
│  - Handle all database operations                        │
│  - Manage transactions                                   │
│  - Handle errors and retries                             │
│  - Validate inputs                                       │
│  - Log operations                                        │
└─────────────────────────────────────────────────────────┘
```

## Search Query Execution Plan

### Exact Match Only
```
User Query: gender="male" AND age_min >= 25
    │
    ▼
Database Query:
    SELECT DISTINCT file_id
    FROM file_attribute
    WHERE (attribute_key='gender' AND attribute_value='male')
    AND file_id IN (
        SELECT file_id FROM file_attribute
        WHERE attribute_key='age_min'
        AND attribute_value::int >= 25
    )
    │
    ▼
Results (unordered)
```

### Semantic Search Only
```
User Query: "mountain resort skiing"
    │
    ▼
Embed Query Text → query_embedding
    │
    ▼
Database Query:
    SELECT file_id,
           1 - (embedding <=> query_embedding) as similarity
    FROM file_embedding
    WHERE embedding_source = 'image'
    ORDER BY embedding <=> query_embedding
    LIMIT 50
    │
    ▼
Results (ordered by similarity)
```

### Combined Search (FILTER_THEN_SEARCH)
```
User Query: gender="male" AND age_min >= 25 AND semantic:"mountains"
    │
    ├─→ Step 1: Exact match filter
    │       │
    │       ▼
    │   filtered_file_ids = [id1, id2, id3, ...]
    │
    └─→ Step 2: Embed query
            │
            ▼
        query_embedding
            │
            ▼
        Database Query:
            SELECT file_id, similarity
            FROM file_embedding
            WHERE file_id IN (id1, id2, id3, ...)
            ORDER BY embedding <=> query_embedding
            LIMIT 50
            │
            ▼
        Results (filtered + ranked)
```

## Deployment Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Load Balancer                            │
└────────────────────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌──────────────┐       ┌──────────────┐
│  Web Server  │       │  Web Server  │
│  (FastAPI)   │       │  (FastAPI)   │
│              │       │              │
│  - Search    │       │  - Search    │
│  - Submit    │       │  - Submit    │
│    Jobs      │       │    Jobs      │
└──────────────┘       └──────────────┘
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │   RabbitMQ (Broker)  │
        └──────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌──────────────┐       ┌──────────────┐
│   Celery     │       │   Celery     │
│   Worker 1   │       │   Worker 2   │
│              │       │              │
│  - Plugins   │       │  - Plugins   │
│  - Embedding │       │  - Embedding │
└──────────────┘       └──────────────┘
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │  PostgreSQL          │
        │  + pgvector          │
        │                      │
        │  - Metadata tables   │
        │  - HNSW indexes      │
        └──────────────────────┘
```
