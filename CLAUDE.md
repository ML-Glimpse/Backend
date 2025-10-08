# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

Start the development server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8010
```

## Architecture Overview

This is a FastAPI-based ML backend for photo recommendations using face recognition and FAISS vector similarity search. The application learns user preferences through swipe interactions and provides personalized photo recommendations.

### Core Data Flow

1. **Photo Upload → Face Embedding**: Photos are stored in MongoDB with binary data. Face embeddings (512-dim vectors) are extracted using InsightFace and cached in the photo documents.

2. **FAISS Index Initialization**: On startup, `faiss_service.initialize_index()` builds an in-memory FAISS index from all photo embeddings. Photos without embeddings are processed on-the-fly during initialization.

3. **User Preference Learning**: When a user "likes" a photo (swipe right), the photo's face embedding is incrementally averaged into the user's `avg_embedding`. Formula: `new_avg = (current_avg * count + new_embedding) / (count + 1)`

4. **Recommendation Generation**: User's `avg_embedding` is used to query the FAISS index (inner product similarity) to find the top-K most similar photos. New users get random recommendations.

### Key Architectural Patterns

**Singleton Services**: All services (`faiss_service`, `face_recognition_service`, `user_service`, etc.) use the singleton pattern - instantiated once at module level and imported elsewhere. The FAISS index and face model are expensive to initialize and must persist across requests.

**MongoDB Collections**: Two main collections defined in `app/core/database.py`:
- `users`: Stores `username`, `hashed_password`, `embeddings[]` (all liked photo embeddings), `avg_embedding`, `embedding_count`
- `photos`: Stores `filename`, `content_type`, `data` (binary), `embedding` (512-dim float array)

**Lazy Embedding Extraction**: Embeddings are computed lazily:
- During FAISS index initialization for photos without embeddings
- During swipe "like" action if photo lacks an embedding
- Never recomputed once stored in MongoDB

**FAISS Index State**: The `faiss_service` maintains three synchronized data structures:
- `faiss_index`: The FAISS IndexFlatIP (inner product) index
- `photo_ids_list`: List of photo IDs corresponding to index positions
- `embeddings_cache`: Dict mapping photo_id → embedding (currently built but not heavily used)

### Critical Implementation Details

**Environment Configuration**: Uses `pydantic-settings` to load config from `.env`. Required: `MONGO_URI`. The `get_settings()` function is cached via `@lru_cache()` to ensure singleton behavior.

**Database Initialization**: MongoDB client is instantiated at module import time in `app/core/database.py`. Ensure MongoDB is accessible before starting the app or connections will fail immediately.

**Face Model Initialization**: InsightFace model loads on module import in `face_recognition.py`. Uses CPU provider. First run may download model files (~100MB).

**Incremental Average Update**: User preference is maintained as an incremental average (not re-averaging all embeddings on each like). This allows O(1) updates but means the average embedding cannot be "unlearned" from specific photos.

**FAISS Index Rebuild**: Index is built once on startup. If new photos are added to MongoDB while the server is running, call `POST /admin/rebuild_index` to refresh the index. The index is in-memory only and lost on restart.

## MongoDB Schema Reference

### users collection
```python
{
    "username": str,
    "hashed_password": str,  # bcrypt hash
    "embeddings": [list[float], ...],  # all liked photo embeddings
    "avg_embedding": list[float] | None,  # running average
    "embedding_count": int  # number of likes
}
```

### photos collection
```python
{
    "_id": ObjectId,
    "filename": str,
    "content_type": str,
    "data": bytes,  # binary image data
    "embedding": list[float] | None  # 512-dim face embedding
}
```

## Adding New Endpoints

All routes are defined in `app/api/routes.py` and registered with the main app in `app/main.py`. Follow the pattern:
1. Add business logic to appropriate service in `app/services/`
2. Define request/response schemas in `app/models/schemas.py`
3. Add route handler in `routes.py` that calls the service
4. Services should raise `HTTPException` for error handling

## Debugging FAISS Issues

- Check index status: `GET /admin/index_status`
- Rebuild index: `POST /admin/rebuild_index`
- Verify photos have embeddings: Query MongoDB `photos` collection for `{embedding: {$exists: true}}`
- Check logs for "FAISS index initialized with X embeddings" on startup
- If recommendations are random, user likely has `avg_embedding: None`
