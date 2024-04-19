from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017"

connection = MongoClient(MONGO_URI)