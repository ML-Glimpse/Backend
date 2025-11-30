"""Photo management and recommendation service"""
import numpy as np
import faiss
from typing import Optional
from bson import ObjectId
from fastapi import HTTPException, UploadFile
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
    async def upload_photo(file: UploadFile, gender: str) -> dict:
        """
        Upload a new photo to the database

        Args:
            file: Uploaded file
            gender: User-selected gender ('M' or 'F')

        Returns:
            Upload status with photo ID

        Raises:
            HTTPException: If upload fails or invalid file
        """
        try:
            # Validate gender
            if gender not in ['M', 'F']:
                raise HTTPException(status_code=400, detail="Gender must be 'M' or 'F'")

            # Validate file type
            if not file.content_type or not file.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="File must be an image")

            # Read file data
            file_data = await file.read()

            if len(file_data) == 0:
                raise HTTPException(status_code=400, detail="File is empty")

            # Extract face embedding
            result = face_recognition_service.extract_embedding(file_data)

            if result is None:
                # No face detected - still save but without embedding
                photo_doc = {
                    "filename": file.filename or "unknown",
                    "content_type": file.content_type,
                    "data": file_data,
                    "gender": gender  # Use user-selected gender
                }
                insert_result = photos_collection.insert_one(photo_doc)
                logger.warning(f"Photo uploaded without face detection: {insert_result.inserted_id}")
                return {
                    "msg": "Photo uploaded but no face detected",
                    "photo_id": str(insert_result.inserted_id),
                    "face_detected": False,
                    "gender": gender
                }

            # Save photo with embedding and user-selected gender
            photo_doc = {
                "filename": file.filename or "unknown",
                "content_type": file.content_type,
                "data": file_data,
                "embedding": result["embedding"],
                "gender": gender  # Use user-selected gender instead of detected
            }

            insert_result = photos_collection.insert_one(photo_doc)
            photo_id = str(insert_result.inserted_id)

            # Rebuild FAISS index to include new photo
            faiss_service.initialize_index()

            logger.info(f"Photo uploaded successfully: {photo_id}, gender: {gender}")

            return {
                "msg": "Photo uploaded successfully",
                "photo_id": photo_id,
                "face_detected": True,
                "gender": gender
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error uploading photo: {e}")
            raise HTTPException(status_code=500, detail=f"Error uploading photo: {str(e)}")

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
        Get photo recommendations for a user based on preference similarity:
        1. Pre-filter available photos at database level
        2. Use FAISS for similarity-based recommendations
        3. Fall back to all available photos if user has no preferences yet

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

        # === Pre-filter at database level ===
        # Build query to get only available photos
        query = {
            "_id": {"$nin": [ObjectId(pid) for pid in excluded_photo_ids]},
            "embedding": {"$exists": True}
        }
        if gender:
            query["gender"] = gender

        # Get available photos from database
        available_photos = list(photos_collection.find(query, {"_id": 1, "embedding": 1, "filename": 1, "content_type": 1}))
        
        if not available_photos:
            return {
                "recommendations": [],
                "recommendation_type": "no_available_photos"
            }

        logger.info(f"Found {len(available_photos)} available photos after pre-filtering")

        recommendations = []
        k = settings.faiss_recommendations_count
        
        # Build temporary FAISS index with only available photos
        embeddings_list = [p["embedding"] for p in available_photos]
        
        # Check if we have valid embeddings
        if not embeddings_list:
            logger.warning("No valid embeddings found in available photos")
            return {
                "recommendations": [],
                "recommendation_type": "no_valid_embeddings"
            }
        
        available_embeddings = np.vstack(embeddings_list).astype("float32")
        available_ids = [str(p["_id"]) for p in available_photos]
        
        # Create temporary index
        temp_index = faiss.IndexFlatIP(settings.faiss_dimension)
        temp_index.add(available_embeddings)
        
        # Determine query embedding
        if user_avg_embedding:
            # Use user's preference embedding
            query_embedding = np.array(user_avg_embedding, dtype="float32").reshape(1, -1)
            recommendation_type = "personalized"
            logger.info(f"Using personalized recommendations based on {user_data.get('embedding_count', 0)} liked photos")
        else:
            # New user: use average of all available photos as query
            # This will return photos in a neutral order based on their similarity to the dataset center
            query_embedding = np.mean(available_embeddings, axis=0, keepdims=True).astype("float32")
            recommendation_type = "new_user_neutral"
            logger.info("New user: using dataset average as query embedding")
        
        # Search for similar photos
        n_results = min(k, len(available_photos))
        similarities, indices = temp_index.search(query_embedding, n_results)
        
        # Build recommendations
        for rank, (similarity, idx) in enumerate(zip(similarities[0], indices[0]), start=1):
            if idx < len(available_photos):
                photo = available_photos[idx]
                recommendations.append({
                    "photo_id": str(photo["_id"]),
                    "filename": photo.get("filename", "unknown"),
                    "content_type": photo.get("content_type", "image/jpeg"),
                    "similarity": float(similarity),
                    "rank": rank
                })
        
        return {
            "recommendations": recommendations,
            "recommendation_type": recommendation_type,
            "based_on_embeddings": user_data.get("embedding_count", 0) if user_avg_embedding else 0
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
