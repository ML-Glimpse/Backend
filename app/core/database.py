"""Database connection and collections"""
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from .config import get_settings

settings = get_settings()

# MongoDB Client
client: MongoClient = MongoClient(settings.mongo_uri)
db: Database = client[settings.mongo_db_name]

# Collections
users_collection: Collection = db["users"]
photos_collection: Collection = db["photos"]
