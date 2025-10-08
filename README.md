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

- `POST /register` - Register a new user
- `POST /login` - Login user

### User Management

- `GET /users/{username}/recommendations` - Get personalized recommendations
- `GET /users/{username}/embeddings` - Get user's embeddings
- `GET /users/{username}/avg_embedding` - Get user's average embedding
- `POST /users/{username}/embeddings` - Add embedding to user
- `POST /users/{username}/search` - Search user embeddings

### Photos

- `GET /photos/{photo_id}` - Get photo data

### Interactions

- `POST /swipe` - Handle swipe action (like/pass)

### Admin

- `POST /admin/rebuild_index` - Rebuild FAISS index
- `GET /admin/index_status` - Get index status

### Health

- `GET /` - Basic health check
- `GET /health` - Detailed health check

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
