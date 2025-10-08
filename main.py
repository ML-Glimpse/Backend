from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field
from passlib.context import CryptContext
from pymongo import MongoClient
import numpy as np
import faiss
import cv2
import insightface
from typing import List
import io
import tempfile
import os
from bson import ObjectId
import logging

#get the uri from .env (MONGO_URI)
uri = GETENV

client = MongoClient(uri)
db = client["ml"]
users = db["users"]
photos = db["photos"]

face_model = insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'])
face_model.prepare(ctx_id=0, det_size=(640, 640))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
app = FastAPI()

faiss_index = None
photo_ids_list = []
embeddings_cache = {}

class User(BaseModel):
    username: str = Field(..., min_length=3)
    password: str = Field(..., min_length=6)

class EmbeddingIn(BaseModel):
    embedding: list[float]

class SearchIn(BaseModel):
    query: list[float]

class SwipeAction(BaseModel):
    username: str
    photo_id: str
    action: str

def initialize_faiss_index():
    """初始化 FAISS 索引，預先計算所有照片的 embeddings"""
    global faiss_index, photo_ids_list, embeddings_cache
    
    print("Initializing FAISS index...")
    
    photos_without_embedding = list(photos.find({"embedding": {"$exists": False}}))
    
    print(f"Found {len(photos_without_embedding)} photos without embeddings")
    
    for photo in photos_without_embedding:
        try:
            nparr = np.frombuffer(photo["data"], np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                faces = face_model.get(img)
                if len(faces) > 0:
                    embedding = faces[0].normed_embedding.astype("float32")
                    embedding_list = embedding.tolist()
                    
                    photos.update_one(
                        {"_id": photo["_id"]},
                        {"$set": {"embedding": embedding_list}}
                    )
                    print(f"Added embedding for photo {photo['_id']}")
        except Exception as e:
            print(f"Error processing photo {photo['_id']}: {e}")
    
    photos_with_embedding = list(photos.find({"embedding": {"$exists": True}}))
    
    if not photos_with_embedding:
        print("No photos with embeddings found")
        return
    
    print(f"Building FAISS index with {len(photos_with_embedding)} photos")
    
    embeddings_list = []
    photo_ids_list = []
    
    for photo in photos_with_embedding:
        embeddings_list.append(np.array(photo["embedding"], dtype="float32"))
        photo_ids_list.append(str(photo["_id"]))
        embeddings_cache[str(photo["_id"])] = photo["embedding"]
    
    if embeddings_list:
        embeddings_array = np.vstack(embeddings_list)
        
        dimension = embeddings_array.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss_index.add(embeddings_array)
        
        print(f"FAISS index initialized with {faiss_index.ntotal} embeddings")
    else:
        print("No valid embeddings found")

def get_photo_recommendations_faiss(user_avg_embedding, k=10):
    """使用 FAISS 獲取最相似的 k 張照片"""
    global faiss_index, photo_ids_list
    
    if faiss_index is None or len(photo_ids_list) == 0:
        return []
    
    query_embedding = np.array(user_avg_embedding, dtype="float32").reshape(1, -1)
    
    actual_k = min(k, faiss_index.ntotal)
    
    similarities, indices = faiss_index.search(query_embedding, actual_k)
    
    recommendations = []
    for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
        if idx < len(photo_ids_list):
            photo_id = photo_ids_list[idx]
            
            photo_info = photos.find_one(
                {"_id": ObjectId(photo_id)}, 
                {"filename": 1, "content_type": 1}
            )
            
            if photo_info:
                recommendations.append({
                    "photo_id": photo_id,
                    "filename": photo_info.get("filename", "unknown"),
                    "content_type": photo_info.get("content_type", "image/jpeg"),
                    "similarity": float(similarity),
                    "rank": i + 1
                })
    
    return recommendations

@app.on_event("startup")
async def startup_event():
    initialize_faiss_index()

@app.post("/register")
def register(user: User):
    if users.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="User already exists")
    hashed = pwd_context.hash(user.password)
    users.insert_one({
        "username": user.username, 
        "hashed_password": hashed, 
        "embeddings": [],
        "avg_embedding": None,
        "embedding_count": 0
    })
    return {"msg": "User registered"}

@app.post("/login")
def login(user: User):
    user_data = users.find_one({"username": user.username})
    if not user_data:
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    if pwd_context.verify(user.password, user_data["hashed_password"]):
        return {"msg": "Login successful"}
    raise HTTPException(status_code=400, detail="Invalid credentials")

@app.get("/users/{username}/recommendations")
def get_recommendations(username: str):
    """使用 FAISS 獲取基於使用者偏好的 10 張推薦照片"""
    global faiss_index, photo_ids_list
    
    user_data = users.find_one({"username": username})
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_avg_embedding = user_data.get("avg_embedding")
    
    if user_avg_embedding and faiss_index is not None:
        recommendations = get_photo_recommendations_faiss(user_avg_embedding, k=10)
        
        return {
            "recommendations": recommendations,
            "recommendation_type": "personalized",
            "based_on_embeddings": user_data.get("embedding_count", 0)
        }
    else:
        if not photo_ids_list:
            return {"recommendations": [], "recommendation_type": "no_photos"}
        
        import random
        random_photo_ids = random.sample(photo_ids_list, min(10, len(photo_ids_list)))
        
        recommendations = []
        for photo_id in random_photo_ids:
            photo_info = photos.find_one(
                {"_id": ObjectId(photo_id)}, 
                {"filename": 1, "content_type": 1}
            )
            if photo_info:
                recommendations.append({
                    "photo_id": photo_id,
                    "filename": photo_info.get("filename", "unknown"),
                    "content_type": photo_info.get("content_type", "image/jpeg"),
                    "similarity": None,
                    "rank": None
                })
        
        return {
            "recommendations": recommendations,
            "recommendation_type": "random" if not user_avg_embedding else "no_index"
        }

@app.get("/photos/{photo_id}")
def get_photo(photo_id: str):
    """獲取照片的二進位內容"""
    try:
        photo = photos.find_one({"_id": ObjectId(photo_id)})
        if not photo:
            raise HTTPException(status_code=404, detail="Photo not found")
        
        return {
            "photo_id": photo_id,
            "filename": photo["filename"],
            "content_type": photo["content_type"],
            "data": photo["data"]
        }
    except:
        raise HTTPException(status_code=400, detail="Invalid photo ID")

@app.post("/swipe")
async def handle_swipe(swipe: SwipeAction):
    """處理使用者的滑動動作"""
    if swipe.action == "like":
        try:
            photo = photos.find_one({"_id": ObjectId(swipe.photo_id)})
            if not photo:
                raise HTTPException(status_code=404, detail="Photo not found")
            
            if "embedding" in photo:
                embedding_list = photo["embedding"]
            else:
                nparr = np.frombuffer(photo["data"], np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    raise HTTPException(status_code=400, detail="Invalid image format")
                
                faces = face_model.get(img)
                
                if len(faces) == 0:
                    return {"msg": "No face detected in the image", "embedding_updated": False}
                
                embedding = faces[0].normed_embedding
                embedding_list = embedding.tolist()
                
                photos.update_one(
                    {"_id": ObjectId(swipe.photo_id)},
                    {"$set": {"embedding": embedding_list}}
                )
            
            user_data = users.find_one({"username": swipe.username})
            if not user_data:
                raise HTTPException(status_code=404, detail="User not found")
            
            current_avg = user_data.get("avg_embedding")
            current_count = user_data.get("embedding_count", 0)
            
            if current_avg is None:
                new_avg = embedding_list
                new_count = 1
            else:
                current_avg_np = np.array(current_avg)
                new_embedding_np = np.array(embedding_list)
                new_avg_np = (current_avg_np * current_count + new_embedding_np) / (current_count + 1)
                new_avg = new_avg_np.tolist()
                new_count = current_count + 1
            
            users.update_one(
                {"username": swipe.username},
                {
                    "$set": {
                        "avg_embedding": new_avg,
                        "embedding_count": new_count
                    },
                    "$push": {"embeddings": embedding_list}
                }
            )
            
            return {
                "msg": "Like recorded and embedding updated",
                "embedding_updated": True,
                "face_detected": True,
                "embedding_count": new_count
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    else:
        return {"msg": "Pass recorded", "embedding_updated": False}

@app.post("/admin/rebuild_index")
def rebuild_faiss_index():
    """重建 FAISS 索引（管理員功能）"""
    try:
        initialize_faiss_index()
        return {
            "msg": "FAISS index rebuilt successfully",
            "total_photos": len(photo_ids_list) if photo_ids_list else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rebuilding index: {str(e)}")

@app.get("/admin/index_status")
def get_index_status():
    """獲取 FAISS 索引狀態"""
    global faiss_index, photo_ids_list
    
    return {
        "index_initialized": faiss_index is not None,
        "total_vectors": faiss_index.ntotal if faiss_index else 0,
        "total_photo_ids": len(photo_ids_list),
        "dimension": 512 if faiss_index else None
    }

@app.post("/users/{username}/embeddings")
def add_embedding(username: str, emb: EmbeddingIn):
    result = users.update_one(
        {"username": username},
        {"$push": {"embeddings": emb.embedding}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"msg": "Embedding added"}

@app.get("/users/{username}/embeddings")
def get_embeddings(username: str):
    user = users.find_one({"username": username}, {"_id": 0, "embeddings": 1})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/users/{username}/avg_embedding")
def get_avg_embedding(username: str):
    """獲取使用者的平均 embedding"""
    user = users.find_one(
        {"username": username}, 
        {"_id": 0, "avg_embedding": 1, "embedding_count": 1}
    )
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.post("/users/{username}/search")
def search_embeddings(username: str, query: SearchIn):
    user = users.find_one({"username": username}, {"_id": 0, "embeddings": 1})
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
