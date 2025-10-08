"""FastAPI application entry point"""
from fastapi import FastAPI
import logging

from app.core.config import get_settings
from app.api.routes import router
from app.services.faiss_service import faiss_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()

# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    debug=settings.debug
)

# Include API routes
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup"""
    logger.info("Starting application...")
    faiss_service.initialize_index()
    logger.info("Application started successfully")


@app.get("/")
def root():
    """Health check endpoint"""
    return {"status": "ok", "app": settings.app_name}


@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "faiss_index": faiss_service.get_index_status()
    }
