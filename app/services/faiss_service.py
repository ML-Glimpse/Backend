"""FAISS indexing and similarity search service"""
import numpy as np
import faiss
from typing import Optional
from bson import ObjectId
import logging

from app.core.database import photos_collection
from app.core.config import get_settings
from .face_recognition import face_recognition_service

logger = logging.getLogger(__name__)
settings = get_settings()


class FAISSService:
    """Service for FAISS vector similarity search"""

    def __init__(self):
        """Initialize FAISS service"""
        self.faiss_index: Optional[faiss.Index] = None
        self.photo_ids_list: list[str] = []
        self.embeddings_cache: dict[str, list[float]] = {}

    def initialize_index(self) -> None:
        """Initialize FAISS index with all photo embeddings"""
        logger.info("Initializing FAISS index...")

        # Process photos without embeddings
        photos_without_embedding = list(photos_collection.find({"embedding": {"$exists": False}}))
        logger.info(f"Found {len(photos_without_embedding)} photos without embeddings")

        for photo in photos_without_embedding:
            try:
                embedding = face_recognition_service.extract_embedding(photo["data"])

                if embedding:
                    photos_collection.update_one(
                        {"_id": photo["_id"]},
                        {"$set": {"embedding": embedding}}
                    )
                    logger.info(f"Added embedding for photo {photo['_id']}")
            except Exception as e:
                logger.error(f"Error processing photo {photo['_id']}: {e}")

        # Build FAISS index
        photos_with_embedding = list(photos_collection.find({"embedding": {"$exists": True}}))

        if not photos_with_embedding:
            logger.warning("No photos with embeddings found")
            return

        logger.info(f"Building FAISS index with {len(photos_with_embedding)} photos")

        embeddings_list = []
        self.photo_ids_list = []
        invalid_count = 0
        expected_dim = settings.faiss_dimension

        for photo in photos_with_embedding:
            try:
                embedding = photo.get("embedding")

                # Skip if embedding is None or empty
                if embedding is None or not embedding:
                    logger.warning(f"Photo {photo['_id']} has null/empty embedding - deleting")
                    photos_collection.delete_one({"_id": photo["_id"]})
                    invalid_count += 1
                    continue

                # Validate embedding dimension
                embedding_array = np.array(embedding, dtype="float32")
                if embedding_array.shape[0] != expected_dim:
                    logger.warning(f"Photo {photo['_id']} has invalid embedding dimension {embedding_array.shape[0]} (expected {expected_dim}) - deleting")
                    photos_collection.delete_one({"_id": photo["_id"]})
                    invalid_count += 1
                    continue

                # Add valid embedding
                embeddings_list.append(embedding_array)
                self.photo_ids_list.append(str(photo["_id"]))
                self.embeddings_cache[str(photo["_id"])] = embedding

            except Exception as e:
                logger.error(f"Error processing embedding for photo {photo['_id']}: {e} - deleting")
                photos_collection.delete_one({"_id": photo["_id"]})
                invalid_count += 1

        if invalid_count > 0:
            logger.info(f"Deleted {invalid_count} photos with invalid embeddings")

        if embeddings_list:
            embeddings_array = np.vstack(embeddings_list)

            dimension = embeddings_array.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(embeddings_array)

            logger.info(f"FAISS index initialized with {self.faiss_index.ntotal} embeddings")
        else:
            logger.warning("No valid embeddings found")

    def get_recommendations(
        self,
        user_avg_embedding: list[float],
        k: int = 10,
        excluded_photo_ids: list[str] = None
    ) -> list[dict]:
        """
        Get photo recommendations based on user's average embedding

        Args:
            user_avg_embedding: User's average preference embedding
            k: Number of recommendations to return
            excluded_photo_ids: List of photo IDs to exclude from recommendations

        Returns:
            List of recommended photos with similarity scores
        """
        if self.faiss_index is None or len(self.photo_ids_list) == 0:
            return []

        if excluded_photo_ids is None:
            excluded_photo_ids = []

        excluded_set = set(excluded_photo_ids)
        logger.info(f"FAISS filtering {len(excluded_set)} excluded photos")
        logger.info(f"Excluded set sample: {list(excluded_set)[:3]}")
        logger.info(f"Photo IDs list sample: {self.photo_ids_list[:3]}")

        query_embedding = np.array(user_avg_embedding, dtype="float32").reshape(1, -1)

        # Request more candidates to account for filtered results
        search_k = min(k * 3, self.faiss_index.ntotal)

        similarities, indices = self.faiss_index.search(query_embedding, search_k)

        recommendations = []
        rank = 1
        filtered_count = 0
        for similarity, idx in zip(similarities[0], indices[0]):
            if len(recommendations) >= k:
                break

            if idx < len(self.photo_ids_list):
                photo_id = self.photo_ids_list[idx]

                # Skip excluded photos
                if photo_id in excluded_set:
                    filtered_count += 1
                    logger.info(f"Filtering out photo_id: {photo_id}")
                    continue

                photo_info = photos_collection.find_one(
                    {"_id": ObjectId(photo_id)},
                    {"filename": 1, "content_type": 1}
                )

                if photo_info:
                    recommendations.append({
                        "photo_id": photo_id,
                        "filename": photo_info.get("filename", "unknown"),
                        "content_type": photo_info.get("content_type", "image/jpeg"),
                        "similarity": float(similarity),
                        "rank": rank
                    })
                    rank += 1

        logger.info(f"Filtered {filtered_count} photos, returning {len(recommendations)} recommendations")
        return recommendations

    def get_index_status(self) -> dict:
        """Get current FAISS index status"""
        return {
            "index_initialized": self.faiss_index is not None,
            "total_vectors": self.faiss_index.ntotal if self.faiss_index else 0,
            "total_photo_ids": len(self.photo_ids_list),
            "dimension": settings.faiss_dimension if self.faiss_index else None
        }


# Global instance
faiss_service = FAISSService()
