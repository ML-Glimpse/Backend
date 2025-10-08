"""Vector search utilities"""
import numpy as np
import faiss
from fastapi import HTTPException

from app.core.database import users_collection
from app.models.schemas import SearchIn


def search_user_embeddings(username: str, query: SearchIn) -> dict:
    """
    Search user's embeddings using FAISS

    Args:
        username: Username
        query: Search query embedding

    Returns:
        Top k similar embeddings with distances

    Raises:
        HTTPException: If user not found or no embeddings stored
    """
    user = users_collection.find_one({"username": username}, {"_id": 0, "embeddings": 1})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    embeddings = user.get("embeddings", [])
    if not embeddings:
        raise HTTPException(status_code=400, detail="No embeddings stored for this user")

    data = np.array(embeddings, dtype="float32")
    q = np.array(query.query, dtype="float32").reshape(1, -1)

    d = data.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(data)

    k = min(5, len(embeddings))
    distances, indices = index.search(q, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            "index": int(idx),
            "distance": float(dist),
            "vector": embeddings[idx]
        })

    return {"results": results}
