"""Pydantic models for request/response validation"""
from pydantic import BaseModel, Field
from typing import Optional


class User(BaseModel):
    """User registration and login schema"""
    username: str = Field(..., min_length=3)
    password: str = Field(..., min_length=6)


class EmbeddingIn(BaseModel):
    """Input schema for embedding data"""
    embedding: list[float]


class SearchIn(BaseModel):
    """Input schema for search queries"""
    query: list[float]


class SwipeAction(BaseModel):
    """Schema for user swipe actions"""
    username: str
    photo_id: str
    action: str


class PhotoRecommendation(BaseModel):
    """Schema for photo recommendation response"""
    photo_id: str
    filename: str
    content_type: str
    similarity: Optional[float] = None
    rank: Optional[int] = None


class RecommendationResponse(BaseModel):
    """Schema for recommendations API response"""
    recommendations: list[PhotoRecommendation]
    recommendation_type: str
    based_on_embeddings: Optional[int] = None
