"""User management service"""
import numpy as np
from typing import Optional
from fastapi import HTTPException
import logging

from app.core.database import users_collection
from app.core.security import hash_password, verify_password
from app.models.schemas import User

logger = logging.getLogger(__name__)


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
            "embedding_count": 0
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
    def update_user_embedding(username: str, embedding: list[float]) -> dict:
        """
        Update user's average embedding with a new liked photo embedding

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
            current_avg_np = np.array(current_avg)
            new_embedding_np = np.array(embedding)
            new_avg_np = (current_avg_np * current_count + new_embedding_np) / (current_count + 1)
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
                    "embeddings": []
                }
            }
        )

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="User not found")

        logger.info(f"Cleared preferences for user {username}")
        return {"msg": "User preferences cleared successfully"}


# Global instance
user_service = UserService()
