"""Face recognition and embedding extraction service"""
import numpy as np
import cv2
import insightface
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FaceRecognitionService:
    """Service for face detection and embedding extraction"""

    def __init__(self):
        """Initialize the face recognition model"""
        self.face_model = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
        self.face_model.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("Face recognition model initialized")

    def extract_embedding(self, image_data: bytes) -> Optional[dict]:
        """
        Extract face embedding and gender from image data

        Args:
            image_data: Binary image data

        Returns:
            Dict with embedding and gender, or None if no face detected
        """
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                logger.warning("Failed to decode image")
                return None

            faces = self.face_model.get(img)

            if len(faces) == 0:
                logger.info("No face detected in image")
                return None

            face = faces[0]
            embedding = face.normed_embedding.astype("float32").tolist()
            # InsightFace: 1 for Male, 0 for Female
            gender = 'M' if face.gender == 1 else 'F'

            return {
                "embedding": embedding,
                "gender": gender
            }

        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None


# Global instance
face_recognition_service = FaceRecognitionService()
