# ML Backend API

A FastAPI-based machine learning backend for photo recommendation using face recognition and FAISS vector similarity search.

## Features

- 🔐 User authentication (registration/login)
- 📸 Face detection and embedding extraction
- 🎯 Personalized photo recommendations using FAISS
- 👍 Swipe-based preference learning
- ⚡ Fast similarity search with vector indexing

## Project Structure

```
Backend/
├── app/
│   ├── api/              # API routes and endpoints
│   │   ├── routes.py     # Main route definitions
│   │   └── __init__.py
│   ├── core/             # Core configuration
│   │   ├── config.py     # Settings and environment variables
│   │   ├── database.py   # MongoDB connection
│   │   ├── security.py   # Password hashing utilities
│   │   └── __init__.py
│   ├── models/           # Data models
│   │   ├── schemas.py    # Pydantic models
│   │   └── __init__.py
│   ├── services/         # Business logic
│   │   ├── face_recognition.py  # Face detection service
│   │   ├── faiss_service.py     # FAISS indexing
│   │   ├── user_service.py      # User management
│   │   ├── photo_service.py     # Photo operations
│   │   ├── swipe_service.py     # Swipe handling
│   │   └── __init__.py
│   ├── utils/            # Helper functions
│   │   ├── search.py     # Search utilities
│   │   └── __init__.py
│   ├── main.py           # Application entry point
│   └── __init__.py
├── .env.example          # Example environment variables
├── .gitignore           # Git ignore rules
├── pyproject.toml       # Project configuration
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Installation

### Prerequisites

- Python 3.10+
- MongoDB
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Backend
```

2. Create and activate a virtual environment:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your MongoDB URI and settings
```

5. Run the application:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8010
```

## Environment Variables

Create a `.env` file with the following variables:

```env
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=ml
APP_NAME=ML Backend API
DEBUG=false
FAISS_DIMENSION=512
FAISS_RECOMMENDATIONS_COUNT=10
```

## API Endpoints

### Authentication

#### `POST /register`
Register a new user account.

**Request Body:**
```json
{
  "username": "string (min 3 characters)",
  "password": "string (min 6 characters)"
}
```

**Response:**
```json
{
  "msg": "User registered"
}
```

**Error Cases:**
- `400 Bad Request` - User already exists
- `422 Unprocessable Entity` - Invalid request body (username too short, password too short)

**Implementation Details:**
- Passwords are hashed using bcrypt before storage
- Creates a new user document with empty embeddings array and null avg_embedding
- Initializes embedding_count to 0

---

#### `POST /login`
Authenticate an existing user.

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "msg": "Login successful"
}
```

**Error Cases:**
- `400 Bad Request` - Invalid credentials (user not found or password incorrect)

**Implementation Details:**
- Uses bcrypt to verify password against stored hash
- Does not issue JWT tokens (stateless authentication not implemented)

---

### User Management

#### `GET /users/{username}/recommendations`
Get personalized photo recommendations based on user's preference history.

**Path Parameters:**
- `username` - The username to get recommendations for

**Response (Personalized):**
```json
{
  "recommendations": [
    {
      "photo_id": "string (MongoDB ObjectId)",
      "filename": "string",
      "content_type": "string (e.g., image/jpeg)",
      "similarity": 0.95,
      "rank": 1
    }
  ],
  "recommendation_type": "personalized",
  "based_on_embeddings": 15
}
```

**Response (Random - New User):**
```json
{
  "recommendations": [
    {
      "photo_id": "string",
      "filename": "string",
      "content_type": "string",
      "similarity": null,
      "rank": null
    }
  ],
  "recommendation_type": "random"
}
```

**Error Cases:**
- `404 Not Found` - User not found

**Implementation Details:**
- If user has avg_embedding (has liked photos), uses FAISS index to find top-K similar photos
- Similarity is calculated using inner product (cosine similarity on normalized embeddings)
- New users without preferences receive random photo recommendations
- Number of recommendations controlled by `FAISS_RECOMMENDATIONS_COUNT` env variable (default: 10)
- See `app/services/photo_service.py:48`

---

#### `GET /users/{username}/embeddings`
Retrieve all face embeddings the user has liked.

**Path Parameters:**
- `username` - The username to get embeddings for

**Response:**
```json
{
  "embeddings": [
    [0.123, -0.456, ...],
    [0.789, -0.321, ...]
  ]
}
```

**Error Cases:**
- `404 Not Found` - User not found

**Implementation Details:**
- Returns raw 512-dimensional face embedding vectors
- Each embedding corresponds to a liked photo
- Used primarily for debugging and analysis

---

#### `GET /users/{username}/avg_embedding`
Get the user's average face embedding (preference profile).

**Path Parameters:**
- `username` - The username to get average embedding for

**Response:**
```json
{
  "avg_embedding": [0.123, -0.456, ...],
  "embedding_count": 15
}
```

**Response (New User):**
```json
{
  "avg_embedding": null,
  "embedding_count": 0
}
```

**Error Cases:**
- `404 Not Found` - User not found

**Implementation Details:**
- The avg_embedding is calculated incrementally: `new_avg = (current_avg * count + new_embedding) / (count + 1)`
- This vector represents the user's aggregated facial preferences
- Used as the query vector for FAISS similarity search
- See `app/services/user_service.py:71`

---

#### `POST /users/{username}/embeddings`
Manually add an embedding to a user's collection (for testing/admin purposes).

**Path Parameters:**
- `username` - The username to add embedding to

**Request Body:**
```json
{
  "embedding": [0.123, -0.456, ...]
}
```

**Response:**
```json
{
  "msg": "Embedding added"
}
```

**Error Cases:**
- `404 Not Found` - User not found

**Implementation Details:**
- Adds embedding to the embeddings array but does NOT update avg_embedding
- This is different from the swipe "like" action which updates the average
- Primarily used for testing and debugging

---

#### `POST /users/{username}/search`
Search the user's liked photo embeddings using FAISS.

**Path Parameters:**
- `username` - The username to search embeddings for

**Request Body:**
```json
{
  "query": [0.123, -0.456, ...]
}
```

**Response:**
```json
{
  "results": [
    {
      "embedding": [0.123, -0.456, ...],
      "similarity": 0.92,
      "rank": 1
    }
  ]
}
```

**Error Cases:**
- `404 Not Found` - User not found
- `400 Bad Request` - No embeddings to search

**Implementation Details:**
- Creates a temporary FAISS index from user's liked embeddings
- Searches for embeddings similar to the query vector
- Used for analyzing user preferences and finding similar liked photos

---

### Photos

#### `GET /photos/{photo_id}`
Retrieve a photo's data including binary content.

**Path Parameters:**
- `photo_id` - MongoDB ObjectId of the photo

**Response:**
```json
{
  "photo_id": "string",
  "filename": "string",
  "content_type": "string",
  "data": "binary data"
}
```

**Error Cases:**
- `404 Not Found` - Photo not found
- `400 Bad Request` - Invalid photo ID format

**Implementation Details:**
- Returns the raw binary image data stored in MongoDB
- Photos also contain cached embeddings if faces have been detected
- Binary data is stored directly in MongoDB (suitable for small-medium datasets)

---

### Interactions

#### `POST /swipe`
Process a user's swipe action on a photo (like or pass).

**Request Body:**
```json
{
  "username": "string",
  "photo_id": "string (MongoDB ObjectId)",
  "action": "like" | "pass"
}
```

**Response (Like with Face Detected):**
```json
{
  "msg": "Like recorded and embedding updated",
  "embedding_updated": true,
  "face_detected": true,
  "embedding_count": 16
}
```

**Response (Pass):**
```json
{
  "msg": "Pass recorded",
  "embedding_updated": false
}
```

**Response (No Face Detected):**
```json
{
  "msg": "No face detected in the image",
  "embedding_updated": false
}
```

**Error Cases:**
- `404 Not Found` - Photo or user not found
- `500 Internal Server Error` - Face detection/processing error

**Implementation Details:**
- "pass" actions are logged but do not update user preferences
- "like" actions trigger face embedding extraction (if not cached)
- Face embeddings are extracted using InsightFace (ArcFace model)
- User's avg_embedding is updated incrementally with the new face embedding
- If photo lacks cached embedding, it's computed on-the-fly and saved to photo document
- This is the primary mechanism for learning user preferences
- See `app/services/swipe_service.py:18`

---

### Admin

#### `POST /admin/rebuild_index`
Rebuild the FAISS similarity search index from all photos in the database.

**Request Body:** None

**Response:**
```json
{
  "msg": "FAISS index rebuilt successfully",
  "total_photos": 1523
}
```

**Error Cases:**
- `500 Internal Server Error` - Error during index rebuild

**Implementation Details:**
- Loads all photos from MongoDB and extracts embeddings if missing
- Creates a new FAISS IndexFlatIP (inner product) index with all photo embeddings
- Index is in-memory only and lost on server restart
- Use this endpoint after uploading new photos to include them in recommendations
- Index is automatically built on application startup
- See `app/services/faiss_service.py` for indexing logic

---

#### `GET /admin/index_status`
Get the current status and statistics of the FAISS index.

**Request Body:** None

**Response:**
```json
{
  "index_initialized": true,
  "total_photos_indexed": 1523,
  "index_dimension": 512
}
```

**Implementation Details:**
- Shows whether FAISS index is ready to serve recommendations
- total_photos_indexed indicates how many photos are searchable
- index_dimension should always be 512 (InsightFace embedding size)
- Use this to verify the index is properly initialized after startup or rebuild

---

### Health

#### `GET /`
Basic health check endpoint.

**Request Body:** None

**Response:**
```json
{
  "status": "ok",
  "app": "ML Backend API"
}
```

**Implementation Details:**
- Always returns 200 OK if the server is running
- Used for basic liveness checks

---

#### `GET /health`
Detailed health check including service status.

**Request Body:** None

**Response:**
```json
{
  "status": "healthy",
  "faiss_index": {
    "index_initialized": true,
    "total_photos_indexed": 1523,
    "index_dimension": 512
  }
}
```

**Implementation Details:**
- Provides comprehensive health information
- Includes FAISS index status for verifying recommendation service readiness
- Used for readiness checks in production deployments

## Development

### Code Formatting

```bash
pip install -e ".[dev]"
black app/
ruff check app/
```

### Type Checking

```bash
mypy app/
```

## Architecture

### Key Components

1. **Face Recognition Service**: Uses InsightFace for face detection and embedding extraction
2. **FAISS Service**: Manages vector similarity search index for fast recommendations
3. **User Service**: Handles user registration, authentication, and preference management
4. **Photo Service**: Manages photo storage and retrieval
5. **Swipe Service**: Processes user interactions and updates preferences

### Recommendation Algorithm

1. User likes a photo (swipe right)
2. Face embedding is extracted from the photo
3. User's average embedding is updated incrementally
4. FAISS index searches for similar photos based on user's preferences
5. Top-K most similar photos are returned as recommendations

## Technologies

- **FastAPI**: Modern, fast web framework
- **MongoDB**: Document database for user and photo data
- **InsightFace**: State-of-the-art face recognition
- **FAISS**: Fast similarity search library by Meta
- **OpenCV**: Image processing
- **Pydantic**: Data validation
- **Passlib**: Password hashing

## License

MIT License
