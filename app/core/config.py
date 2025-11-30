"""Application configuration settings"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # MongoDB Configuration
    mongo_uri: str
    mongo_db_name: str = "ml"

    # Application Settings
    app_name: str = "ML Backend API"
    debug: bool = False

    # FAISS Settings
    faiss_dimension: int = 512
    faiss_recommendations_count: int = 10

    # Recommendation Algorithm Settings
    negative_feedback_weight: float = 0.02  # Weight for dislike embedding push-away
    super_like_weight_multiplier: float = 3.0  # Super like has 3x stronger effect on preferences
    
    # Dynamic Learning Rate Settings
    early_learning_decay_min: float = 0.5  # Fast learning for first 10 likes
    early_learning_decay_max: float = 0.7  # Transition rate at 10 likes
    mid_learning_decay: float = 0.9  # Stable learning at 50+ likes
    early_learning_threshold: int = 10  # Number of likes to switch from early to mid learning
    stable_learning_threshold: int = 50  # Number of likes to reach stable learning

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
