"""API route definitions"""
from fastapi import APIRouter
from fastapi.responses import Response

from app.models.schemas import User, EmbeddingIn, SearchIn, SwipeAction
from app.services.user_service import user_service
from app.services.photo_service import photo_service
from app.services.swipe_service import swipe_service
from app.services.faiss_service import faiss_service
from app.utils.search import search_user_embeddings

router = APIRouter()


# Authentication endpoints
@router.post("/register")
def register(user: User):
    """Register a new user"""
    return user_service.register_user(user)


@router.post("/login")
def login(user: User):
    """Authenticate user login"""
    return user_service.login_user(user)


# User endpoints
@router.post("/users/{username}/embeddings")
def add_embedding(username: str, emb: EmbeddingIn):
    """Add an embedding to user's collection"""
    return user_service.add_embedding(username, emb.embedding)


@router.get("/users/{username}/embeddings")
def get_embeddings(username: str):
    """Get all user embeddings"""
    return user_service.get_user_embeddings(username)


@router.get("/users/{username}/avg_embedding")
def get_avg_embedding(username: str):
    """Get user's average embedding"""
    return user_service.get_user_avg_embedding(username)


@router.delete("/users/{username}/preferences")
def clear_preferences(username: str):
    """Clear user's preference embeddings and average"""
    return user_service.clear_user_preferences(username)


@router.post("/users/{username}/search")
def search_embeddings(username: str, query: SearchIn):
    """Search user's embeddings using FAISS"""
    return search_user_embeddings(username, query)


@router.get("/users/{username}/recommendations")
def get_recommendations(username: str):
    """Get personalized photo recommendations for user"""
    return photo_service.get_recommendations(username)


# Photo endpoints
@router.get("/photos/{photo_id}")
def get_photo(photo_id: str):
    """Get photo binary data"""
    photo_data = photo_service.get_photo(photo_id)
    return Response(
        content=photo_data["data"],
        media_type=photo_data["content_type"]
    )


# Swipe endpoints
@router.post("/swipe")
async def handle_swipe(swipe: SwipeAction):
    """Handle user swipe action (like/pass)"""
    return swipe_service.handle_swipe(swipe)


# Admin endpoints
@router.post("/admin/process_embeddings")
def process_all_embeddings(force: bool = False):
    """Process all photos and add embeddings (admin only)"""
    return photo_service.process_all_photo_embeddings(force=force)


@router.post("/admin/rebuild_index")
def rebuild_faiss_index():
    """Rebuild FAISS index (admin only)"""
    try:
        faiss_service.initialize_index()
        return {
            "msg": "FAISS index rebuilt successfully",
            "total_photos": len(faiss_service.photo_ids_list)
        }
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Error rebuilding index: {str(e)}")


@router.get("/admin/index_status")
def get_index_status():
    """Get FAISS index status"""
    return faiss_service.get_index_status()
