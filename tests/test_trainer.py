import pytest
from unittest.mock import MagicMock, patch, mock_open
import pandas as pd
import numpy as np

from film_recommender_cg.trainer import Trainer

@pytest.fixture
def mock_db_connector():
    connector = MagicMock()
    
    ratings_data = pd.DataFrame({
        'user_id': [1, 1, 2, 2, 3, 3],
        'film_id': [10, 11, 10, 12, 11, 12],
        'rating': [4, 5, 3, 2, 5, 4],
        'original_data': [True, True, True, True, True, True]
    })
    connector.read_collection.return_value = ratings_data
    return connector

@pytest.fixture
def trainer_instance():
    mock_db_connector = MagicMock()
    mock_db_connector.read_collection.side_effect = lambda name: (
        pd.DataFrame({
            'user_id': [1, 1, 2],
            'film_id': [101, 102, 103],
            'rating': [4, 5, 3],
            'original_data': [True, True, True]
        }) if name == 'ratings' else
        pd.DataFrame({
            'film_id': [101, 102, 103],
            'genres': [
                "[{'id': 1, 'name': 'Action'}, {'id': 2, 'name': 'Comedy'}]",
                "[{'id': 1, 'name': 'Action'}, {'id': 2, 'name': 'Comedy'}]",
                "[{'id': 1, 'name': 'Action'}, {'id': 2, 'name': 'Comedy'}]"
            ]
        })
    )

    mock_wandb_run = MagicMock()   # ✅ mock run so .use_artifact exists

    with patch("wandb.login") as mock_login, \
         patch("wandb.init", return_value=mock_wandb_run) as mock_init, \
         patch("wandb.log") as mock_log, \
         patch("wandb.finish") as mock_finish:

        mock_login.return_value = None
        mock_log.return_value = None
        mock_finish.return_value = None

        trainer = Trainer(
            db_connector=mock_db_connector,
            new_user_threshold=0,
            wandb_auth_key="fake_key",
            wandb_project_path="fake_project_path",
            wandb_project_name="fake_project_name"
        )

    yield trainer

def test_load_ratings_data(trainer_instance):
    ratings, ratings_existing = trainer_instance._load_ratings_data("2020-01-01", "2020-12-31")
    assert isinstance(ratings, pd.DataFrame)
    assert isinstance(ratings_existing, pd.DataFrame)
    assert 'user_id' in ratings.columns

def test_train_svd_model(trainer_instance):
    trainer_instance._load_ratings_data("2020-01-01", "2020-12-31")
    model = trainer_instance._train_svd_model()
    assert model is not None
    assert hasattr(trainer_instance, "trained_model")
    assert hasattr(trainer_instance, "testset")

def test_evaluate_model_performance(trainer_instance):
    trainer_instance._load_ratings_data("2020-01-01", "2020-12-31")
    trainer_instance._train_svd_model()

    with patch.object(trainer_instance, "_log_artifact") as mock_log_artifact, \
         patch("wandb.log") as mock_wandb_log:
        metrics = trainer_instance._evaluate_model_performance()
        assert "Overall" in metrics
        assert "By Genre" in metrics

def test_upload_trained_model(trainer_instance):
    trainer_instance._load_ratings_data("2020-01-01", "2020-12-31")
    trainer_instance._train_svd_model()

    with patch("wandb.Artifact") as mock_artifact_class, \
         patch("wandb.log_artifact") as mock_log_artifact, \
         patch("wandb.log") as mock_wandb_log, \
         patch("os.remove") as mock_os_remove:
        trainer_instance._upload_trained_model("prod")
        mock_artifact_class.assert_called()
        mock_log_artifact.assert_called()
        mock_wandb_log.assert_called()

def test_import_current_champion_model(trainer_instance):
    with patch("os.path.join", return_value="fake_path"), \
         patch("builtins.open", new_callable=mock_open) as mock_file, \
         patch("pickle.load", return_value=MagicMock()) as mock_pickle, \
         patch.object(trainer_instance.wandb_run, "use_artifact", return_value=MagicMock(download=lambda: ".")):
        model = trainer_instance._import_current_champion_model()
        assert model is not None

def test_get_degraded_metrics(trainer_instance):
    # Patch _access_artifact to return fake metrics
    fake_metrics = {
        'Overall': {'r_squared': 0.9},
        'By Genre': {'Action': {'r_squared': 0.8}, 'Comedy': {'r_squared': 0.95}}
    }
    with patch.object(trainer_instance, "_access_artifact", return_value=fake_metrics):
        degraded, stable = trainer_instance._get_degraded_metrics()
        assert isinstance(degraded, dict)
        assert isinstance(stable, dict)

def test_get_metric_comparison_outcomes(trainer_instance):
    champion_metrics = {'Overall': {'r_squared': 0.8}, 'By Genre': {'Action': {'r_squared': 0.7}}}
    challenger_metrics = {'Overall': {'r_squared': 0.85}, 'By Genre': {'Action': {'r_squared': 0.72}}}
    relevant_metrics = ['Overall', 'Action']
    
    outcomes = trainer_instance._get_metric_comparison_outcomes(champion_metrics, challenger_metrics, relevant_metrics, 'degraded')
    assert 'Overall' in outcomes
    assert 'By Genre' in outcomes

def test_notify_team(trainer_instance):
    # Patch smtplib to avoid sending real email
    with patch("smtplib.SMTP") as mock_smtp:
        trainer_instance._notify_team(
            subject="Test",
            message="This is a test",
            recipients="test@example.com",
            sender_email="sender@example.com",
            sender_password="password"
        )
        mock_smtp.assert_called()
