"""Photo management and recommendation service"""
import random
from typing import Optional
from bson import ObjectId
from fastapi import HTTPException
import logging

from app.core.database import photos_collection, users_collection
from app.core.config import get_settings
from .faiss_service import faiss_service

logger = logging.getLogger(__name__)
settings = get_settings()


class PhotoService:
    """Service for photo management and recommendations"""

    @staticmethod
    def get_photo(photo_id: str) -> dict:
        """
        Get photo data by ID

        Args:
            photo_id: Photo ID

        Returns:
            Photo data including binary content

        Raises:
            HTTPException: If photo not found or invalid ID
        """
        try:
            photo = photos_collection.find_one({"_id": ObjectId(photo_id)})
            if not photo:
                raise HTTPException(status_code=404, detail="Photo not found")

            return {
                "photo_id": photo_id,
                "filename": photo["filename"],
                "content_type": photo["content_type"],
                "data": photo["data"]
            }
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid photo ID")

    @staticmethod
    def get_recommendations(username: str) -> dict:
        """
        Get photo recommendations for a user

        Args:
            username: Username

        Returns:
            Recommended photos with metadata

        Raises:
            HTTPException: If user not found
        """
        user_data = users_collection.find_one({"username": username})
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")

        user_avg_embedding = user_data.get("avg_embedding")

        # Personalized recommendations if user has preferences
        if user_avg_embedding and faiss_service.faiss_index is not None:
            recommendations = faiss_service.get_recommendations(
                user_avg_embedding,
                k=settings.faiss_recommendations_count
            )

            return {
                "recommendations": recommendations,
                "recommendation_type": "personalized",
                "based_on_embeddings": user_data.get("embedding_count", 0)
            }
        else:
            # Random recommendations for new users
            if not faiss_service.photo_ids_list:
                return {"recommendations": [], "recommendation_type": "no_photos"}

            random_photo_ids = random.sample(
                faiss_service.photo_ids_list,
                min(settings.faiss_recommendations_count, len(faiss_service.photo_ids_list))
            )

            recommendations = []
            for photo_id in random_photo_ids:
                photo_info = photos_collection.find_one(
                    {"_id": ObjectId(photo_id)},
                    {"filename": 1, "content_type": 1}
                )
                if photo_info:
                    recommendations.append({
                        "photo_id": photo_id,
                        "filename": photo_info.get("filename", "unknown"),
                        "content_type": photo_info.get("content_type", "image/jpeg"),
                        "similarity": None,
                        "rank": None
                    })

            return {
                "recommendations": recommendations,
                "recommendation_type": "random" if not user_avg_embedding else "no_index"
            }


# Global instance
photo_service = PhotoService()
