"""User management service"""
import numpy as np
from typing import Optional
from fastapi import HTTPException
import logging

from app.core.database import users_collection
from app.core.security import hash_password, verify_password
from app.models.schemas import User
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class UserService:
    """Service for user management operations"""

    @staticmethod
    def register_user(user: User) -> dict:
        """
        Register a new user

        Args:
            user: User registration data

        Returns:
            Success message

        Raises:
            HTTPException: If user already exists
        """
        if users_collection.find_one({"username": user.username}):
            raise HTTPException(status_code=400, detail="User already exists")

        hashed = hash_password(user.password)
        users_collection.insert_one({
            "username": user.username,
            "hashed_password": hashed,
            "embeddings": [],
            "avg_embedding": None,
            "embedding_count": 0,
            "liked_photos": [],
            "disliked_photos": []
        })

        logger.info(f"User {user.username} registered successfully")
        return {"msg": "User registered"}

    @staticmethod
    def login_user(user: User) -> dict:
        """
        Authenticate user login

        Args:
            user: User login credentials

        Returns:
            Success message

        Raises:
            HTTPException: If credentials are invalid
        """
        user_data = users_collection.find_one({"username": user.username})
        if not user_data:
            raise HTTPException(status_code=400, detail="Invalid credentials")

        if verify_password(user.password, user_data["hashed_password"]):
            logger.info(f"User {user.username} logged in successfully")
            return {"msg": "Login successful"}

        raise HTTPException(status_code=400, detail="Invalid credentials")

    @staticmethod
    def _calculate_dynamic_decay(embedding_count: int) -> float:
        """
        Calculate dynamic decay rate based on user's embedding count
        
        Early stage (0-10 likes): Fast learning (0.5-0.7)
        Mid stage (10-50 likes): Gradual stabilization (0.7-0.9)
        Stable stage (50+ likes): High stability (0.9)
        
        Args:
            embedding_count: Current number of liked embeddings
            
        Returns:
            Decay rate (0-1)
        """
        if embedding_count < settings.early_learning_threshold:
            # Early learning: 0.5 -> 0.7
            progress = embedding_count / settings.early_learning_threshold
            decay = settings.early_learning_decay_min + progress * (
                settings.early_learning_decay_max - settings.early_learning_decay_min
            )
        elif embedding_count < settings.stable_learning_threshold:
            # Mid learning: 0.7 -> 0.9
            progress = (embedding_count - settings.early_learning_threshold) / (
                settings.stable_learning_threshold - settings.early_learning_threshold
            )
            decay = settings.early_learning_decay_max + progress * (
                settings.mid_learning_decay - settings.early_learning_decay_max
            )
        else:
            # Stable learning: 0.9
            decay = settings.mid_learning_decay
        
        logger.debug(f"Embedding count: {embedding_count}, decay rate: {decay:.3f}")
        return decay

    @staticmethod
    def update_user_embedding(username: str, embedding: list[float]) -> dict:
        """
        Update user's average embedding with a new liked photo embedding
        Uses dynamic decay rate based on user's experience level

        Args:
            username: Username
            embedding: New face embedding to add

        Returns:
            Update status with new embedding count

        Raises:
            HTTPException: If user not found
        """
        user_data = users_collection.find_one({"username": username})
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")

        current_avg = user_data.get("avg_embedding")
        current_count = user_data.get("embedding_count", 0)

        if current_avg is None:
            new_avg = embedding
            new_count = 1
        else:
            # Calculate dynamic decay rate
            decay = UserService._calculate_dynamic_decay(current_count)
            
            current_avg_np = np.array(current_avg)
            new_embedding_np = np.array(embedding)

            # Exponential Moving Average with dynamic decay
            # NewAvg = (OldAvg * Decay) + (NewPhoto * (1 - Decay))
            new_avg_np = (current_avg_np * decay) + (new_embedding_np * (1.0 - decay))
            
            # CRITICAL: Re-normalize to unit vector
            # FAISS IndexFlatIP requires unit vectors for valid Cosine Similarity
            norm = np.linalg.norm(new_avg_np)
            if norm > 0:
                new_avg_np = new_avg_np / norm
                
            new_avg = new_avg_np.tolist()
            new_count = current_count + 1

        users_collection.update_one(
            {"username": username},
            {
                "$set": {
                    "avg_embedding": new_avg,
                    "embedding_count": new_count
                },
                "$push": {"embeddings": embedding}
            }
        )

        logger.info(f"Updated embedding for user {username}, count: {new_count}")

        return {
            "msg": "Embedding updated",
            "embedding_count": new_count
        }

    @staticmethod
    def update_user_embedding_negative(username: str, embedding: list[float]) -> dict:
        """
        Update user's average embedding by pushing away from disliked photo
        Uses negative feedback to refine user preferences

        Args:
            username: Username
            embedding: Face embedding from disliked photo

        Returns:
            Update status

        Raises:
            HTTPException: If user not found
        """
        user_data = users_collection.find_one({"username": username})
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")

        current_avg = user_data.get("avg_embedding")
        
        # Only apply negative feedback if user has established preferences
        if current_avg is None:
            logger.info(f"User {username} has no preferences yet, skipping negative feedback")
            return {"msg": "No preferences to update", "applied": False}
        
        current_avg_np = np.array(current_avg)
        disliked_embedding_np = np.array(embedding)
        
        # Push away from disliked embedding
        # new_avg = current_avg - (disliked * weight)
        new_avg_np = current_avg_np - (disliked_embedding_np * settings.negative_feedback_weight)
        
        # Re-normalize to unit vector
        norm = np.linalg.norm(new_avg_np)
        if norm > 0:
            new_avg_np = new_avg_np / norm
        else:
            # If resulting vector is zero, keep original
            logger.warning(f"Negative feedback resulted in zero vector for {username}, keeping original")
            return {"msg": "Negative feedback skipped (zero vector)", "applied": False}
        
        new_avg = new_avg_np.tolist()
        
        users_collection.update_one(
            {"username": username},
            {"$set": {"avg_embedding": new_avg}}
        )
        
        logger.info(f"Applied negative feedback for user {username}")
        
        return {
            "msg": "Negative feedback applied",
            "applied": True
        }

    @staticmethod
    def get_user_embeddings(username: str) -> dict:
        """Get all embeddings for a user"""
        user = users_collection.find_one({"username": username}, {"_id": 0, "embeddings": 1})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user

    @staticmethod
    def get_user_avg_embedding(username: str) -> dict:
        """Get user's average embedding"""
        user = users_collection.find_one(
            {"username": username},
            {"_id": 0, "avg_embedding": 1, "embedding_count": 1}
        )
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user

    @staticmethod
    def add_embedding(username: str, embedding: list[float]) -> dict:
        """Add an embedding to user's collection"""
        result = users_collection.update_one(
            {"username": username},
            {"$push": {"embeddings": embedding}}
        )
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="User not found")
        return {"msg": "Embedding added"}

    @staticmethod
    def get_swiped_photos(username: str) -> dict:
        """
        Get list of all photos the user has swiped (both liked and disliked)

        Args:
            username: Username

        Returns:
            Dict with liked_photos and disliked_photos arrays

        Raises:
            HTTPException: If user not found
        """
        user = users_collection.find_one(
            {"username": username},
            {"_id": 0, "liked_photos": 1, "disliked_photos": 1}
        )
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return {
            "liked_photos": user.get("liked_photos", []),
            "disliked_photos": user.get("disliked_photos", [])
        }

    @staticmethod
    def clear_user_preferences(username: str) -> dict:
        """
        Clear user's preference embeddings and reset average

        Args:
            username: Username

        Returns:
            Success message

        Raises:
            HTTPException: If user not found
        """
        result = users_collection.update_one(
            {"username": username},
            {
                "$set": {
                    "avg_embedding": None,
                    "embedding_count": 0,
                    "embeddings": [],
                    "liked_photos": [],
                    "disliked_photos": []
                }
            }
        )

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="User not found")

        logger.info(f"Cleared preferences for user {username}")
        return {"msg": "User preferences cleared successfully"}


# Global instance
user_service = UserService()
