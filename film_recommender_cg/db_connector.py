"""
db_connector.py
"""

import logging
import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo.collection import Collection
from pymongo.database import Database
from typing import Optional, Union, List, Dict, Any

logger_db_connector = logging.getLogger(__name__)

class DBConnector:
    """
    Handles MongoDB database connections, queries, insertions, and deletions.
    """

    def __init__(self, uri: str, db_name: str) -> None:
        """
        Initialize the DBConnector with a MongoDB URI and database name.

        Args:
            uri (str): MongoDB connection URI.
            db_name (str): Name of the database to connect to.
        """
        self.uri = uri
        self.db_name = db_name
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None

    def get_connection(self) -> Database:
        """
        Establish a connection to the MongoDB database.

        Returns:
            Database: The connected MongoDB database instance.
        """
        try:
            self.client = MongoClient(self.uri, server_api=ServerApi('1'))
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]

            logger_db_connector.info("Successfully connected to DB.")
            return self.db

        except Exception:
            logger_db_connector.exception("Error occurred in establishing DB connection.")
            raise

    def read_collection(
        self,
        collection_name: str,
        query: Optional[Dict[str, Any]] = None,
        sort_on: Optional[Union[str, List[str]]] = None,
        exclude_id: bool = True
    ) -> pd.DataFrame:
        """
        Read data from a MongoDB collection into a pandas DataFrame.

        Args:
            collection_name (str): Name of the collection to read from.
            query (Optional[Dict[str, Any]]): MongoDB query filter. Defaults to None.
            sort_on (Optional[Union[str, List[str]]]): Field(s) to sort on. Defaults to None.
            exclude_id (bool): Whether to exclude the '_id' field. Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame containing the query results.
        """
        try:
            if self.db is None:
                self.get_connection()

            collection: Collection = self.db[collection_name]
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

    def insert_documents(
        self,
        collection_name: str,
        documents: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> None:
        """
        Insert one or more documents into a MongoDB collection.

        Args:
            collection_name (str): Name of the collection to insert into.
            documents (Union[Dict[str, Any], List[Dict[str, Any]]]): Document(s) to insert.
        """
        try:
            if self.db is None:
                self.get_connection()

            collection: Collection = self.db[collection_name]

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

    def delete_documents(self, collection_name: str, filter_query: Dict[str, Any]) -> None:
        """
        Delete one or more documents from a MongoDB collection based on a filter query.

        Args:
            collection_name (str): Name of the collection to delete from.
            filter_query (Dict[str, Any]): Query filter specifying which documents to delete.
        """
        try:
            if self.db is None:
                self.get_connection()

            collection: Collection = self.db[collection_name]
            result = collection.delete_many(filter_query)

            logger_db_connector.info(
                f"Deleted {result.deleted_count} document(s) from '{collection_name}'."
            )

        except Exception:
            logger_db_connector.exception(f"Error occurred while deleting from '{collection_name}'.")
            raise

    def reset_ratings(self) -> None:
        """
        Delete all non-original (i.e., simulated) data from the 'ratings' collection.
        """
        self.delete_documents('ratings', {"original_data": False})

    def close_connection(self) -> None:
        """
        Close the MongoDB client connection if it exists.
        """
        try:
            if self.client:
                self.client.close()
                logger_db_connector.info("DB connection closed.")
        except Exception:
            logger_db_connector.exception("Error occurred while closing DB connection.")
            raise