"""Photo management and recommendation service"""
import random
from typing import Optional
from bson import ObjectId
from fastapi import HTTPException
import logging

from app.core.database import photos_collection, users_collection
from app.core.config import get_settings
from .faiss_service import faiss_service
from .face_recognition import face_recognition_service

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

        # Get list of photos to exclude (already swiped)
        liked_photos = user_data.get("liked_photos", [])
        disliked_photos = user_data.get("disliked_photos", [])
        excluded_photo_ids = set(liked_photos + disliked_photos)

        logger.info(f"User {username}: {len(liked_photos)} liked, {len(disliked_photos)} disliked")
        logger.info(f"Excluded IDs sample: {list(excluded_photo_ids)[:3]}")

        # Personalized recommendations if user has preferences
        if user_avg_embedding and faiss_service.faiss_index is not None:
            recommendations = faiss_service.get_recommendations(
                user_avg_embedding,
                k=settings.faiss_recommendations_count,
                excluded_photo_ids=list(excluded_photo_ids)
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

            # Filter out already swiped photos
            available_photo_ids = [
                photo_id for photo_id in faiss_service.photo_ids_list
                if photo_id not in excluded_photo_ids
            ]

            if not available_photo_ids:
                return {
                    "recommendations": [],
                    "recommendation_type": "all_photos_swiped"
                }

            random_photo_ids = random.sample(
                available_photo_ids,
                min(settings.faiss_recommendations_count, len(available_photo_ids))
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

    @staticmethod
    def process_all_photo_embeddings(force: bool = False) -> dict:
        """
        Process all photos in the database and add embeddings

        Args:
            force: If True, re-process all photos even if they have embeddings

        Returns:
            Statistics about the processing operation
        """
        logger.info(f"Starting to process photos for embeddings (force={force})")

        # Find photos to process
        if force:
            photos_to_process = list(photos_collection.find({}))
            logger.info(f"Force mode: processing all {len(photos_to_process)} photos")
        else:
            photos_to_process = list(photos_collection.find({
                "$or": [
                    {"embedding": {"$exists": False}},
                    {"embedding": None}
                ]
            }))
            logger.info(f"Found {len(photos_to_process)} photos without embeddings")

        total_photos = photos_collection.count_documents({})

        logger.info(f"Total photos in database: {total_photos}")

        processed_count = 0
        failed_count = 0
        no_face_count = 0
        deleted_count = 0

        for photo in photos_to_process:
            try:
                embedding = face_recognition_service.extract_embedding(photo["data"])

                if embedding:
                    photos_collection.update_one(
                        {"_id": photo["_id"]},
                        {"$set": {"embedding": embedding}}
                    )
                    processed_count += 1
                    logger.info(f"Added embedding for photo {photo['_id']}")
                else:
                    # Delete photo if no face detected
                    photos_collection.delete_one({"_id": photo["_id"]})
                    deleted_count += 1
                    no_face_count += 1
                    logger.warning(f"No face detected in photo {photo['_id']} - deleted from database")

            except Exception as e:
                failed_count += 1
                logger.error(f"Error processing photo {photo['_id']}: {e}")

        photos_with_embedding = photos_collection.count_documents({"embedding": {"$exists": True}})
        remaining_photos = photos_collection.count_documents({})

        return {
            "msg": "Photo embedding processing completed",
            "total_photos": total_photos,
            "photos_processed": processed_count,
            "photos_with_embeddings": photos_with_embedding,
            "no_face_detected": no_face_count,
            "photos_deleted": deleted_count,
            "remaining_photos": remaining_photos,
            "failed": failed_count
        }


# Global instance
photo_service = PhotoService()
