"""
db_connector.py
"""

import logging
import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

logger_db_connector = logging.getLogger('db_connector')  # TODO: change back to __name__

class DBConnector:

    def __init__(self, uri, db_name):
        self.uri = uri
        self.db_name = db_name
        self.client = None
        self.db = None

    def get_connection(self):
        try:
            self.client = MongoClient(self.uri, server_api=ServerApi('1'))
            self.client.admin.command('ping')  # verify connection
            self.db = self.client[self.db_name]

            logger_db_connector.info("Successfully connected to DB.")
            return self.db

        except Exception:
            logger_db_connector.exception("Error occurred in establishing DB connection.")
            raise

    def read_collection(self, collection_name, query=None, sort_on=None, exclude_id=True):
        try:
            if self.db is None:
                self.get_connection()

            collection = self.db[collection_name]

            projection = {"_id": 0} if exclude_id else None

            cursor = collection.find(query or {}, projection)

            if sort_on:
                cursor = cursor.sort(sort_on)

            df = pd.DataFrame(list(cursor))
            logger_db_connector.info(
                f"Successfully read collection '{collection_name}'. Shape of df: {df.shape}"
            )
            return df

        except Exception:
            logger_db_connector.exception(f"Error occurred in reading collection '{collection_name}'.")
            raise

    def insert_documents(self, collection_name, documents):
        try:
            if self.db is None:
                self.get_connection()

            collection = self.db[collection_name]

            if isinstance(documents, list):
                result = collection.insert_many(documents)
                logger_db_connector.info(
                    f"Inserted {len(result.inserted_ids)} documents into '{collection_name}'."
                )
            else:
                result = collection.insert_one(documents)
                logger_db_connector.info(
                    f"Inserted 1 document into '{collection_name}' (id={result.inserted_id})."
                )

        except Exception:
            logger_db_connector.exception(f"Error occurred while inserting into '{collection_name}'.")
            raise

    def delete_documents(self, collection_name, filter_query):
        try:
            if self.db is None:
                self.get_connection()

            collection = self.db[collection_name]
            result = collection.delete_many(filter_query)

            logger_db_connector.info(
                f"Deleted {result.deleted_count} document(s) from '{collection_name}'."
            )

        except Exception:
            logger_db_connector.exception(f"Error occurred while deleting from '{collection_name}'.")
            raise

    def reset_ratings(self):
        self.delete_documents('ratings', {"original_data": False})

    def close_connection(self):
        try:
            if self.client:
                self.client.close()
                logger_db_connector.info("DB connection closed.")
        except Exception:
            logger_db_connector.exception("Error occurred while closing DB connection.")
            raise
