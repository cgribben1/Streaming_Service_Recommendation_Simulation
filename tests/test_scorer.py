import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from film_recommender.scorer import Scorer

@pytest.fixture
def mock_db_connector():
    """Mock DB connector."""
    return MagicMock()

@pytest.fixture
def scorer_instance(mock_db_connector):
    """Instantiate a Scorer with mocked W&B login/init and DB connector."""
    with patch("scorer.wandb.login"), patch("scorer.wandb.init") as mock_init:
        mock_run = MagicMock()
        mock_init.return_value = mock_run
        scorer = Scorer(
            db_connector=mock_db_connector,
            wandb_auth_key="fake_key",
            wandb_project_path="/fake/path",
            wandb_project_name="test_project",
            new_user_threshold=5,
            n_recs=10,
            max_interactions_between_scorings=3
        )
    return scorer

def test_close_wandb_run(scorer_instance):
    """Test closing W&B run."""
    scorer_instance.close_wandb_run()
    scorer_instance.wandb_run.finish.assert_called_once()

def test_score_simulate_and_update_ratings_table(scorer_instance):
    """Test the main scoring function with all dependencies mocked."""
    # Mock private methods
    mock_ratings = pd.DataFrame({"user_id": [1, 2], "film_id": [101, 102], "rating": [5, 4]})
    mock_rating_counts = pd.DataFrame({"film_id": [101, 102], "count": [10, 15]})
    mock_existing_users = [1]
    mock_new_users = [2]
    mock_all_users = [1, 2]
    mock_ratings_existing = mock_ratings[mock_ratings["user_id"].isin(mock_existing_users)]
    mock_ratings_new = mock_ratings[mock_ratings["user_id"].isin(mock_new_users)]
    mock_genres_table = pd.DataFrame({"film_id": [101, 102], "genre": ["Action", "Comedy"]})
    mock_model = MagicMock()

    scorer_instance._load_ratings_data = MagicMock(return_value=(
        mock_ratings,
        mock_rating_counts,
        mock_ratings_existing,
        mock_ratings_new,
        mock_all_users,
        mock_existing_users,
        mock_new_users
    ))
    scorer_instance._load_genres_table = MagicMock(return_value=mock_genres_table)
    scorer_instance._import_model = MagicMock(return_value=mock_model)
    scorer_instance._get_ranked_films_per_user_existing = MagicMock(return_value=pd.DataFrame({"user_id": [1], "film_id": [101]}))
    scorer_instance._get_ranked_films_per_user_new = MagicMock(return_value=pd.DataFrame({"user_id": [2], "film_id": [102]}))
    scorer_instance._simulate_batch_of_film_choices = MagicMock(side_effect=[
        pd.DataFrame({"user_id": [1], "film_id": [101], "rating": [5]}),
        pd.DataFrame({"user_id": [2], "film_id": [102], "rating": [4]})
    ])
    scorer_instance._update_ratings_table = MagicMock(return_value=pd.DataFrame({
        "user_id": [1, 2],
        "film_id": [101, 102],
        "rating": [5, 4]
    }))

    # Run the method
    scorer_instance.score_simulate_and_update_ratings_table()

    # Assertions
    assert scorer_instance.ratings.equals(mock_ratings)
    assert scorer_instance.rating_counts.equals(mock_rating_counts)
    assert scorer_instance.rankings_generated_existing_users is not None
    assert scorer_instance.rankings_generated_new_users is not None
    assert not scorer_instance.rankings_generated_all_users.empty
    assert scorer_instance.new_interactions_existing_users is not None
    assert scorer_instance.new_interactions_new_users is not None
    assert scorer_instance.new_interactions_all_users is not None

def test_wandb_run_closed_on_exception(mock_db_connector):
    """Ensure W&B run is closed if an exception occurs."""
    with patch("scorer.wandb.login"), patch("scorer.wandb.init") as mock_init:
        mock_run = MagicMock()
        mock_init.return_value = mock_run
        scorer = Scorer(
            db_connector=mock_db_connector,
            wandb_auth_key="fake_key",
            wandb_project_path="/fake/path",
            wandb_project_name="test_project",
            new_user_threshold=5,
            n_recs=10,
            max_interactions_between_scorings=3
        )

    scorer._load_ratings_data = MagicMock(side_effect=Exception("DB error"))

    with pytest.raises(Exception):
        scorer.score_simulate_and_update_ratings_table()

    mock_run.finish.assert_called_once()
