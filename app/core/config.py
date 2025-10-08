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

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
