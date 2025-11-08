"""
test_db_connector.py
"""

import pytest
import pandas as pd
from unittest.mock import patch
import mongomock

from film_recommender.db_connector import DBConnector

@pytest.fixture
def db_connector():
    with patch("film_recommender.db_connector.MongoClient", new=mongomock.MongoClient):
        connector = DBConnector(uri="mongodb://localhost:27017", db_name="test_db")
        connector.get_connection()
        yield connector
        connector.close_connection()

def test_connection(db_connector):
    db = db_connector.get_connection()
    assert db_connector.db is not None
    assert db.name == "test_db"

def test_insert_single_document(db_connector):
    doc = {"name": "Alice", "score": 10}
    db_connector.insert_documents("test_collection", doc)

    df = db_connector.read_collection("test_collection")
    assert len(df) == 1
    assert df.iloc[0]["name"] == "Alice"
    assert df.iloc[0]["score"] == 10

def test_insert_multiple_documents(db_connector):
    docs = [{"name": "Bob"}, {"name": "Charlie"}]
    db_connector.insert_documents("test_collection", docs)

    df = db_connector.read_collection("test_collection")
    assert len(df) == 2
    assert set(df["name"]) == {"Bob", "Charlie"}

def test_read_collection_with_sort(db_connector):
    docs = [{"name": "Charlie", "score": 5}, {"name": "Bob", "score": 10}]
    db_connector.insert_documents("test_collection", docs)

    df = db_connector.read_collection("test_collection", sort_on="score")
    assert df.iloc[0]["score"] == 5
    assert df.iloc[1]["score"] == 10

def test_read_collection_exclude_id(db_connector):
    doc = {"name": "Alice"}
    db_connector.insert_documents("test_collection", doc)

    df = db_connector.read_collection("test_collection", exclude_id=True)
    assert "_id" not in df.columns

def test_delete_documents(db_connector):
    docs = [{"name": "Alice"}, {"name": "Bob"}]
    db_connector.insert_documents("test_collection", docs)

    db_connector.delete_documents("test_collection", {"name": "Alice"})
    df = db_connector.read_collection("test_collection")
    assert len(df) == 1
    assert df.iloc[0]["name"] == "Bob"

def test_reset_ratings(db_connector):
    docs = [
        {"original_data": True, "rating": 5},
        {"original_data": False, "rating": 3}
    ]
    db_connector.insert_documents("ratings", docs)

    db_connector.reset_ratings()
    df = db_connector.read_collection("ratings")
    assert len(df) == 1
    assert df.iloc[0]["original_data"] == True
