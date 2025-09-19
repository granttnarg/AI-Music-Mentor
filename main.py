
from db.db import AudioRAGDatabase
import os
from dotenv import load_dotenv

load_dotenv()

connection_url = os.getenv("DB_CONNECTION_URL")
db = AudioRAGDatabase(connection_url)
db.reset_database()  # Drops all tables
print("db dropped")
db.setup_database()  # Recreates schema
print("db reset")