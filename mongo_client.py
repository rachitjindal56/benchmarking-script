import os
import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)


class MongoDBConfig:
    _instance = None

    def __new__(cls) -> 'MongoDBConfig':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, 'client', None) is None:
            self.client: Optional[AsyncIOMotorClient] = None
            self.mongodb_url = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
            self.database_name = os.getenv('DATABASE_NAME', 'benchmarking')
            self._connected = False

    async def connect_to_database(self) -> bool:
        try:
            self.client = AsyncIOMotorClient(self.mongodb_url, serverSelectionTimeoutMS=5000)
            await self.client.admin.command('ping')
            self._connected = True
            logger.info(f"Connected to MongoDB at {self.mongodb_url}")
            return True
        except Exception as e:
            logger.warning(f"MongoDB not available: {e}")
            self._connected = False
            return False

    async def close_database_connection(self) -> None:
        if self.client is not None:
            self.client.close()
            self._connected = False
            logger.info("MongoDB connection closed")

    def get_database(self):
        if self.client is None:
            self.client = AsyncIOMotorClient(self.mongodb_url, serverSelectionTimeoutMS=5000)
        return self.client[self.database_name]

    @property
    def is_connected(self) -> bool:
        return self._connected


mongo_db = MongoDBConfig()