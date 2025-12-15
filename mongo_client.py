import os
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient


class MongoDBConfig:
    _instance = None
    _is_initialized = False

    def __new__(cls) -> 'MongoDBConfig':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, 'client', None) is None:
            self.client: Optional[AsyncIOMotorClient] = None
            self.mongodb_url = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
            self.database_name = os.getenv('DATABASE_NAME', 'benchmarking')

    async def connect_to_database(self) -> None:
        try:
            self.client = AsyncIOMotorClient(self.mongodb_url)
            await self.client.admin.command('ping')
            print(f"Successfully connected to MongoDB at {self.mongodb_url}")
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")
            raise

    async def close_database_connection(self) -> None:
        if self.client is not None:
            self.client.close()
            print("MongoDB connection closed")

    def get_database(self):
        if self.client is None:
            self.client = AsyncIOMotorClient(self.mongodb_url)
        return self.client[self.database_name]


mongo_db = MongoDBConfig()