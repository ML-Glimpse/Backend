"""Swipe action processing service"""
from bson import ObjectId
from fastapi import HTTPException
import logging

from app.core.database import photos_collection
from app.models.schemas import SwipeAction
from .face_recognition import face_recognition_service
from .user_service import user_service

logger = logging.getLogger(__name__)


class SwipeService:
    """Service for handling user swipe actions"""

    @staticmethod
    def handle_swipe(swipe: SwipeAction) -> dict:
        """
        Process user swipe action (like/pass)

        Args:
            swipe: Swipe action data

        Returns:
            Processing result with embedding update status

        Raises:
            HTTPException: If photo/user not found or processing fails
        """
        from app.core.database import users_collection

        try:
            # Get photo
            photo = photos_collection.find_one({"_id": ObjectId(swipe.photo_id)})
            if not photo:
                raise HTTPException(status_code=404, detail="Photo not found")

            # Handle "pass" action with negative feedback
            if swipe.action == "pass":
                # Record disliked photo
                users_collection.update_one(
                    {"username": swipe.username},
                    {"$addToSet": {"disliked_photos": swipe.photo_id}}
                )

                # Apply negative feedback to user preferences
                # Get or extract embedding
                if "embedding" in photo:
                    embedding_list = photo["embedding"]
                else:
                    result = face_recognition_service.extract_embedding(photo["data"])
                    if result:
                        embedding_list = result["embedding"]
                        gender = result["gender"]

                        # Save embedding to photo
                        photos_collection.update_one(
                            {"_id": ObjectId(swipe.photo_id)},
                            {
                                "$set": {
                                    "embedding": embedding_list,
                                    "gender": gender
                                }
                            }
                        )
                    else:
                        # No face detected, just record the dislike
                        return {
                            "msg": "Pass recorded (no face detected)",
                            "embedding_updated": False
                        }

                # Apply negative feedback
                negative_result = user_service.update_user_embedding_negative(
                    swipe.username,
                    embedding_list
                )

                return {
                    "msg": "Pass recorded with negative feedback",
                    "embedding_updated": negative_result.get("applied", False),
                    "negative_feedback_applied": True
                }

            # Handle "like" and "super_like" actions
            # Get or extract embedding
            if "embedding" in photo:
                embedding_list = photo["embedding"]
            else:
                result = face_recognition_service.extract_embedding(photo["data"])

                if result is None:
                    return {
                        "msg": "No face detected in the image",
                        "embedding_updated": False
                    }
                
                embedding_list = result["embedding"]
                gender = result["gender"]

                # Save embedding to photo
                photos_collection.update_one(
                    {"_id": ObjectId(swipe.photo_id)},
                    {
                        "$set": {
                            "embedding": embedding_list,
                            "gender": gender
                        }
                    }
                )

            # Record liked photo
            users_collection.update_one(
                {"username": swipe.username},
                {"$addToSet": {"liked_photos": swipe.photo_id}}
            )

            # Determine if this is a super like
            if swipe.action == "super_like":
                # Super like: stronger embedding update
                result = user_service.update_user_embedding_super_like(swipe.username, embedding_list)
                return {
                    "msg": "Super like recorded and embedding updated with stronger weight",
                    "embedding_updated": True,
                    "face_detected": True,
                    "embedding_count": result["embedding_count"],
                    "super_like": True
                }
            else:
                # Normal like
                result = user_service.update_user_embedding(swipe.username, embedding_list)
                return {
                    "msg": "Like recorded and embedding updated",
                    "embedding_updated": True,
                    "face_detected": True,
                    "embedding_count": result["embedding_count"]
                }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing swipe: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


# Global instance
swipe_service = SwipeService()
