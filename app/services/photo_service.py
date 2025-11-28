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
    def get_recommendations(username: str, gender: Optional[str] = None) -> dict:
        """
        Get photo recommendations for a user

        Args:
            username: Username
            gender: Optional gender filter ('M' or 'F')

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
            # Request more candidates if filtering by gender to ensure we have enough after filtering
            k = settings.faiss_recommendations_count * 3 if gender else settings.faiss_recommendations_count
            
            recommendations = faiss_service.get_recommendations(
                user_avg_embedding,
                k=k,
                excluded_photo_ids=list(excluded_photo_ids)
            )
            
            # Filter by gender if requested
            if gender:
                filtered_recs = []
                for rec in recommendations:
                    # We need to check the gender from DB
                    photo = photos_collection.find_one(
                        {"_id": ObjectId(rec["photo_id"])},
                        {"gender": 1}
                    )
                    # If gender matches or is not set (to be safe), include it
                    # But user specifically asked for gender filter, so maybe strict?
                    # Let's be strict if gender is present.
                    if photo and photo.get("gender") == gender:
                        filtered_recs.append(rec)
                
                recommendations = filtered_recs[:settings.faiss_recommendations_count]

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
            
            # Filter by gender if requested
            if gender:
                # Query DB for IDs that match gender
                # We convert available_photo_ids to ObjectIds for the query
                available_oids = [ObjectId(pid) for pid in available_photo_ids]
                
                gender_photos = list(photos_collection.find(
                    {
                        "gender": gender, 
                        "_id": {"$in": available_oids}
                    },
                    {"_id": 1}
                ))
                available_photo_ids = [str(p["_id"]) for p in gender_photos]

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
                    {"embedding": None},
                    {"gender": {"$exists": False}}
                ]
            }))
            logger.info(f"Found {len(photos_to_process)} photos without embeddings or gender")

        total_photos = photos_collection.count_documents({})

        logger.info(f"Total photos in database: {total_photos}")

        processed_count = 0
        failed_count = 0
        no_face_count = 0
        deleted_count = 0

        for photo in photos_to_process:
            try:
                result = face_recognition_service.extract_embedding(photo["data"])

                if result:
                    photos_collection.update_one(
                        {"_id": photo["_id"]},
                        {
                            "$set": {
                                "embedding": result["embedding"],
                                "gender": result["gender"]
                            }
                        }
                    )
                    processed_count += 1
                    logger.info(f"Added embedding and gender for photo {photo['_id']}")
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
