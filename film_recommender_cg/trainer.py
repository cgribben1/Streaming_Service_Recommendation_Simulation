'''
trainer.py
'''

import os
import json
import pickle
import logging
import pandas as pd
import numpy as np
from itertools import chain
import wandb
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Any, Dict, List, Tuple, Optional, Union

from film_recommender_cg.db_connector import DBConnector

logger_trainer = logging.getLogger(__name__)


class Trainer:
    """Handles model training, evaluation, retraining logic, and artifact management for the film recommender system."""

    def __init__(
        self,
        db_connector: DBConnector,
        new_user_threshold: int,
        wandb_auth_key: str,
        wandb_project_path: str,
        wandb_project_name: str
    ) -> None:
        """
        Initialize Trainer with database connector and Weights & Biases project details.

        Args:
            db_connector: Database connector instance.
            new_user_threshold: Minimum number of films rated for a user to be considered existing.
            wandb_auth_key: Weights & Biases API key.
            wandb_project_path: Full project path in W&B (e.g., 'user/project').
            wandb_project_name: Name of the W&B project.
        """
        self.db_connector = db_connector
        self.new_user_threshold = new_user_threshold
        self.wandb_auth_key = wandb_auth_key
        self.wandb_project_path = wandb_project_path
        self.wandb_project_name = wandb_project_name
        self.wandb_run = None
        self.data_start_date = None
        self.data_end_date = None
        self.ratings = None
        self.ratings_existing = None
        self.testset = None
        self.trained_model = None
        self.imported_current_champion_model = None

        wandb.login(key=wandb_auth_key)
        self.wandb_run = wandb.init(project=wandb_project_name, job_type='training', tags=["training"])

        logger_trainer.info("'Trainer' object instantiated; W&B run initialized.")

    def close_wandb_run(self) -> None:
        """Close the active Weights & Biases run, if open."""
        if self.wandb_run:
            self.wandb_run.finish()
            logger_trainer.info("Training W&B run closed.")
        else:
            logger_trainer.info("No training W&B run currently open, therefore nothing to close.")

    def train_initial_model(self) -> None:
        """Train the initial SVD model, evaluate it, and upload it to Weights & Biases."""
        try:
            try:
                self._load_ratings_data('start_date', 'end_date')
            except Exception:
                logger_trainer.exception("Error occurred in loading in 'ratings' data.")
                raise

            try:
                self._train_svd_model()
            except Exception:
                logger_trainer.exception("Error occurred in training SVD model.")
                raise

            try:
                self._evaluate_model_performance()
            except Exception:
                logger_trainer.exception("Error occurred in evaluating model performance.")
                raise

            try:
                self._upload_trained_model('prod')
            except Exception:
                logger_trainer.exception("Error occurred in uploading trained model to Weights & Biases.")
                raise

        except Exception:
            logger_trainer.exception("Error occurred during intial model training.")
            raise
        finally:
            self.close_wandb_run()

    def check_if_retraining_required(self) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
        """
        Check whether retraining is required by comparing benchmark and scoring metrics.

        Returns:
            A tuple containing:
                - retraining_required: Whether retraining is necessary.
                - degraded_metrics: Metrics that have degraded.
                - stable_metrics: Metrics that remained stable.
        """
        print("Checking if retraining required...")

        try:
            degraded_metrics, stable_metrics = self._get_degraded_metrics()
            retraining_required = len(degraded_metrics) > 0
            if retraining_required:
                print("Retraining required as at least one performance metric found to have degraded in the current model...")
            else:
                print("No need for retraining as no performance metrics found to have degraded in the current model!")
            return retraining_required, degraded_metrics, stable_metrics
        except Exception:
            logger_trainer.exception("Error occurred in checking if retraining required.")
            self.close_wandb_run()
            raise

    def retrain(self, degraded_metrics: Dict[str, Any], stable_metrics: Dict[str, Any]) -> None:
        """
        Retrain the model if metrics have degraded and evaluate whether the new model should replace the current one.

        Args:
            degraded_metrics: Metrics identified as degraded.
            stable_metrics: Metrics identified as stable.
        """
        try:
            try:
                self._load_ratings_data('start_date', 'end_date')
            except Exception:
                logger_trainer.exception("Error occurred in initializing retrainer object.")
                raise

            try:
                champion = self._import_current_champion_model()
            except Exception:
                logger_trainer.exception("Error occurred in importing current 'champion' model.")
                raise

            try:
                challenger = self._train_svd_model()
                self._evaluate_model_performance()
            except Exception:
                logger_trainer.exception("Error occurred in training and evaluating new 'challenger' model.")
                raise

            try:
                champion_performance_metrics = self._evaluate_model_recent_performance(champion)
            except Exception:
                logger_trainer.exception("Error occurred in evaluating 'champion' model performance metrics on recent data.")
                raise

            try:
                challenger_performance_metrics = self._evaluate_model_recent_performance(challenger)
            except Exception:
                logger_trainer.exception("Error occurred in evaluating 'challenger' model performance metrics on recent data.")
                raise

            try:
                degraded_metric_names = self._get_metric_names(degraded_metrics)
                stable_metric_names = self._get_metric_names(stable_metrics)

                degraded_metric_comparison_outcomes = self._get_metric_comparison_outcomes(
                    champion_performance_metrics, challenger_performance_metrics, degraded_metric_names, 'degraded', tolerance=0.05
                )
                stable_metric_comparison_outcomes = self._get_metric_comparison_outcomes(
                    champion_performance_metrics, challenger_performance_metrics, stable_metric_names, 'stable', tolerance=0.05
                )

                retraining_outcome = self._get_overall_retraining_outcome(degraded_metric_comparison_outcomes, stable_metric_comparison_outcomes)

                if retraining_outcome == 'Promote Challenger':
                    self._upload_trained_model('prod')

                    logger_trainer.info(f"""
                    Retraining was triggered because the performance of {degraded_metric_names} had degraded since the last training.
                    The challenger model improved one or more degraded metrics while keeping the remaining metrics within the allowed tolerance.
                    The challenger has therefore been promoted to production as the new champion model!
                    """)

                elif retraining_outcome == 'Reject Challenger':
                    self._upload_trained_model('staging')

                    logger_trainer.info(f"""
                    Retraining was triggered because the performance of {degraded_metric_names} had degraded since the last training.
                    However, the challenger model failed to improve the degraded metrics within the tolerance threshold.
                    The current champion model remains in production.
                    """)

                elif retraining_outcome == 'Human Review':
                    self._upload_trained_model('staging')

                    logger_trainer.info(f"""
                    Retraining was triggered because the performance of {degraded_metric_names} showed degradation since the last training.
                    The challenger model produced mixed results — some metrics improved while others degraded beyond tolerance.
                    The retraining outcome requires human review before promotion decisions can be made.
                    Please review retraining outcome in Weights & Biases - run information below:
                    Project: https://wandb.ai/{self.wandb_project_path}
                    Run name: {self.wandb_run.name}
                    """)

                    subject = 'Human Review Required for Film Recommender Model Retraining'
                    message = f"""\ 
                    The latest retraining run for the Film Recommender model requires human review.

                    One or more performance metrics degraded during automated evaluation, and the model
                    was not automatically promoted to production.

                    Please review the logged challenger and champion model performance in Weights & Biases
                    to determine whether manual promotion or further tuning is needed.

                    Weights & Biases project: https://wandb.ai/{self.wandb_project_name}
                    Run name: {self.wandb_run.name}
                    """

                    self._notify_team(subject, message, 'curtisgribben1@gmail.com', 'curtisgribben1@gmail.com', 'umrs swlv lxmd pswj')

            except Exception:
                logger_trainer.exception("Error occurred in logic surrounding comparison of 'challenger' vs 'champion' model performance.")
                raise

        finally:
            self.close_wandb_run()

    def _load_ratings_data(self, start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load ratings data from the database and split users into 'existing' and 'new' based on the rating threshold.

        Args:
            start_date: Start date for the data window (placeholder in this demo).
            end_date: End date for the data window (placeholder in this demo).

        Returns:
            A tuple containing:
                - ratings: Complete ratings DataFrame.
                - ratings_existing: Ratings DataFrame filtered for existing users.
        """
        logger_trainer.info("Loading in 'ratings' data...")

        ratings = self.db_connector.read_collection('ratings')

        rating_counts = ratings.groupby('user_id').size().reset_index(name='num_rated_films')
        ratings_with_rating_counts = ratings.merge(rating_counts, on="user_id")

        ratings_existing = ratings_with_rating_counts[ratings_with_rating_counts["num_rated_films"] > self.new_user_threshold].drop('num_rated_films', axis=1).reset_index(drop=True)

        logger_trainer.info("Data successfully loaded!")

        self.data_start_date = start_date
        self.data_end_date = end_date
        self.ratings = ratings
        self.ratings_existing = ratings_existing

        return ratings, ratings_existing

    def _train_svd_model(self) -> SVD:
        """
        Train a Singular Value Decomposition (SVD) model using Surprise on the ratings dataset.

        Returns:
            The trained SVD model instance.
        """
        logger_trainer.info("Training SVD model...")

        reader = Reader(rating_scale=(0, 5))

        data = Dataset.load_from_df(self.ratings_existing[['user_id', 'film_id', 'rating']], reader)
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

        model = SVD(random_state=42)
        model.fit(trainset)

        logger_trainer.info("SVD model successfully trained!")

        self.trained_model = model
        self.testset = testset

        return model

    def _evaluate_model_performance(self) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
        """
        Evaluate model performance both overall and by genre.

        Returns:
            Nested dictionary of performance metrics (RMSE, R²) overall and by genre.
        """
        logger_trainer.info("Evaluating SVD model...")

        model = self.trained_model
        testset = self.testset

        all_performance_metrics = {}
        all_performance_metrics['Overall'] = self._evaluate_performance_metrics_overall(model, testset)
        all_performance_metrics['By Genre'] = self._evaluate_performance_metrics_by_genre(model, testset)

        self._log_artifact("training_performance_metrics", all_performance_metrics)

        logger_trainer.info("Evaluation of SVD model complete!")

        model.performance_metrics = all_performance_metrics

        return all_performance_metrics
    
    def _upload_trained_model(self, environment: str) -> None:
        """
        Upload the trained model to Weights & Biases as an artifact.

        Args:
            environment: The model environment tag ('prod' or 'staging').
        """
        logger_trainer.info(f"Logging SVD model in W&B...")

        model_path = '/tmp/svd_model.pkl'

        with open(model_path, "wb") as f:
            pickle.dump(self.trained_model, f)

        artifact = wandb.Artifact(name="svd_model", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact, aliases=[environment])

        wandb.log({"training_data_start_date": self.data_start_date, "training_data_end_date": self.data_end_date})

        logger_trainer.info("SVD model successfully logged!")

        os.remove(model_path)

    def _log_artifact(self, label: str, data: Dict[str, Any]) -> None:
        """
        Log a JSON artifact to Weights & Biases.

        Args:
            label: Artifact name label.
            data: Data to be serialized and logged.
        """
        tmp_path = f"{label}.json"  

        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=4)

        artifact = wandb.Artifact(label, type="metric")
        artifact.add_file(tmp_path)

        self.wandb_run.log_artifact(artifact)

        os.remove(tmp_path)

        logger_trainer.info(f"'{label}' successfully logged as artifact in W&B.")

    def _access_artifact(self, artifact_name: str) -> Dict[str, Any]:
        """
        Retrieve and load a JSON artifact from Weights & Biases.

        Args:
            artifact_name: Name of the artifact to access.

        Returns:
            The loaded artifact content as a dictionary.
        """
        api = wandb.Api() # no login required as will already be logged in within runtime...

        artifact = api.artifact(f"{os.path.join(self.wandb_project_path, artifact_name)}:latest")
        artifact_dir = artifact.download()
        file_path = os.path.join(artifact_dir, f"{artifact_name}.json")

        with open(file_path, "r") as f:
            loaded_artifact = json.load(f)

        return loaded_artifact

    def _evaluate_performance_metrics_overall(self, model: SVD, testset: List[Tuple[Any, Any, float]]) -> Dict[str, float]:
        """
        Evaluate the model’s overall RMSE and R² performance.

        Args:
            model: Trained SVD model.
            testset: List of testset tuples (user_id, film_id, rating).

        Returns:
            Dictionary with overall RMSE and R².
        """
        predictions = model.test(testset)

        rmse = round(accuracy.rmse(predictions, verbose=False), 3)

        y_true = [pred.r_ui for pred in predictions]
        y_pred = [pred.est for pred in predictions]

        r2 = round(r2_score(y_true, y_pred), 3)

        logger_trainer.info("### Model Performance Metrics (Overall): ###")
        logger_trainer.info(f"RMSE: {rmse}")
        logger_trainer.info(f"squared: {r2}")

        performance_metrics = {'rmse': rmse, 'r_squared': r2}

        wandb.log(performance_metrics) # logging as metric as well as artifact (see 'evaluate_model_performance()' below)

        return performance_metrics
    
    @staticmethod
    def _simplify_genres_list(x: str) -> List[str]:
        """
        Simplify a stringified list of genre dictionaries into a list of genre names.

        Args:
            x: String representation of genre list (e.g., "[{'id': 1, 'name': 'Action'}]").

        Returns:
            List of genre names.
        """
        raw_genres_list = eval(x)
        simplified_genres_list = [raw_genres_list[i]['name'] for i in range(len(raw_genres_list))]
        
        return simplified_genres_list
    
    @staticmethod
    def _fill_in_missing_ids(genres_table: pd.DataFrame) -> pd.DataFrame:
        """
        Fill in missing film IDs in the genres table with placeholder 'Undisclosed' entries.

        Args:
            genres_table: DataFrame containing 'film_id' and 'genres' columns.

        Returns:
            Updated genres DataFrame with all film IDs present.
        """
        full_film_ids = pd.Series(range(1, 5000), name='film_id')
        missing_film_ids = full_film_ids[~full_film_ids.isin(genres_table['film_id'])]

        missing_rows = pd.DataFrame({
            'film_id': missing_film_ids,
            'genres': [['Undisclosed']] * len(missing_film_ids)
        })

        genres_table = pd.concat([genres_table, missing_rows], ignore_index=True)
        genres_table = genres_table.sort_values('film_id').reset_index(drop=True)

        return genres_table
    
    def _load_genres_table(self) -> pd.DataFrame:
        """
        Load and preprocess the genres table from the database.

        Returns:
            Cleaned genres DataFrame with simplified genre lists and missing IDs filled.
        """
        logger_trainer.info("Loading in 'genres' table...")

        genres_table = self.db_connector.read_collection('genres')
        
        genres_table['genres'] = genres_table['genres'].apply(lambda x: self._simplify_genres_list(x))

        genres_table = self._fill_in_missing_ids(genres_table)

        logger_trainer.info("Table successfully loaded!")

        return genres_table 

    def _evaluate_performance_metrics_by_genre(self, model: SVD, testset: List[Tuple[Any, Any, float]]) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Evaluate model performance metrics grouped by genre.

        Args:
            model: Trained SVD model.
            testset: List of testset tuples (user_id, film_id, rating).

        Returns:
            Dictionary mapping genres to RMSE and R² performance metrics.
        """
        predictions = model.test(testset)

        predictions = pd.DataFrame(
            [(pred.uid, pred.iid, pred.r_ui, pred.est) for pred in predictions],
            columns=['user_id', 'film_id', 'rating', 'predicted_rating']
        )

        genres_table = self._load_genres_table()

        predictions_plus_genres = predictions.merge(genres_table, on='film_id', how='left')

        unique_genres = list(set(chain.from_iterable(genres_table['genres'].values)))
        performance_metrics = {}

        logger_trainer.info("### Model Performance Metrics (by Genre): ###")

        for genre in unique_genres:
            genre_relevant_predictions = predictions_plus_genres[predictions_plus_genres['genres'].apply(lambda x: genre in x)]

            if len(genre_relevant_predictions) > 100:
                predicted_ratings = genre_relevant_predictions['predicted_rating'].values
                simmed_ratings = genre_relevant_predictions['rating'].values
                
                rmse = round(np.sqrt(mean_squared_error(simmed_ratings, predicted_ratings)), 3)
                r2 = round(r2_score(simmed_ratings, predicted_ratings), 3)
            
                logger_trainer.info(f"'{genre}' rmse = {rmse}")
                logger_trainer.info(f"'{genre}' r-squared = {r2}")
            else:
                rmse = None
                r2 = None

                logger_trainer.info(f"'{genre}' rmse = Insufficient Data")
                logger_trainer.info(f"'{genre}' r-squared = Insufficient Data")

            performance_metrics[genre] = {'rmse': rmse, 'r_squared': r2}

        return performance_metrics

    def _import_current_champion_model(self) -> SVD:
        """
        Import the current champion SVD model artifact from Weights & Biases.

        Returns:
            The loaded champion SVD model instance.
        """
        logger_trainer.info("Importing 'Champion' SVD model from Weights & Biases...")

        artifact_path = os.path.join(self.wandb_project_path, 'svd_model:latest')
        model_artifact = self.wandb_run.use_artifact(artifact_path, type="model")
        model_dir = model_artifact.download()
        model_file = os.path.join(model_dir, "svd_model.pkl")

        with open(model_file, "rb") as f:
            model = pickle.load(f)

        logger_trainer.info("Model successfully imported!")

        self.imported_current_champion_model = model

        return model

    def _evaluate_model_recent_performance(self, model: SVD) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
        """
        Evaluate a model’s performance on recent ratings data.

        Args:
            model: SVD model to evaluate.

        Returns:
            Nested dictionary containing overall and by-genre performance metrics.
        """
        logger_trainer.info("Evaluating model performance on recent data...")

        reader = Reader(rating_scale=(0, 5))

        recent_ratings = self.ratings_existing[self.ratings_existing['original_data'] == False] # slight workaround due to lack of proper timestamps - returns all 'new' or 'simulated' data (works for this demo as I'm only retraining following one round of scoring)

        testset_df = pd.DataFrame(self.testset, columns=['user_id', 'film_id', 'rating']) # whole of block is just filtering to records in holdout 'testset', plus formatting necessary for scikit-surprise data objects...
        recent_ratings_testset = recent_ratings.merge(testset_df[['user_id', 'film_id']], on=['user_id', 'film_id'], how='inner')
        data = Dataset.load_from_df(recent_ratings_testset[['user_id', 'film_id', 'rating']], reader)
        recent_ratings_testset = data.build_full_trainset().build_testset()

        all_performance_metrics = {}
        all_performance_metrics['Overall'] = self._evaluate_performance_metrics_overall(model, recent_ratings_testset)
        all_performance_metrics['By Genre'] = self._evaluate_performance_metrics_by_genre(model, recent_ratings_testset)

        logger_trainer.info("Evaluation complete!")

        return all_performance_metrics

    @staticmethod
    def _get_metric_names(metric_dict: Dict[str, Any]) -> List[str]:
        """
        Extract all metric names (overall and by genre) from a nested metrics dictionary.

        Args:
            metric_dict: Dictionary of performance metrics.

        Returns:
            List of metric names.
        """
        metric_names = []

        for k, v in metric_dict.items():
            if k == 'Overall':
                metric_names.append(k)

            else:
                metric_names.extend(list(v.keys()))

        return metric_names

    def _get_degraded_metrics(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Compare benchmark and scoring metrics to determine which have degraded or remained stable.

        Returns:
            A tuple containing:
                - degraded_metrics: Metrics that have degraded.
                - stable_metrics: Metrics that remained stable.
        """
        benchmark_performance_metrics = self._access_artifact('training_performance_metrics')['Overall']
        benchmark_performance_metrics_by_genre = self._access_artifact('training_performance_metrics')['By Genre']

        scoring_performance_metrics = self._access_artifact('scoring_performance_metrics_existing_users')['Overall']
        scoring_performance_metrics_by_genre = self._access_artifact('scoring_performance_metrics_existing_users')['By Genre']

        degraded_metrics = {}
        stable_metrics = {}

        overall_benchmark_r2 = benchmark_performance_metrics['r_squared']
        overall_scoring_r2 = scoring_performance_metrics['r_squared']

        overall_r2_degraded = overall_benchmark_r2 - overall_scoring_r2 > 0.02

        if overall_r2_degraded:
            degraded_metrics['Overall'] = {}
            degraded_metrics['Overall']['benchmark_r_squared'] = overall_benchmark_r2
            degraded_metrics['Overall']['scoring_r_squared'] = overall_scoring_r2
        else:
            stable_metrics['Overall'] = {}
            stable_metrics['Overall']['benchmark_r_squared'] = overall_benchmark_r2
            stable_metrics['Overall']['scoring_r_squared'] = overall_scoring_r2

        genres_with_degraded_metric = {}
        genres_with_stable_metric = {}

        for genre, perf_dict in benchmark_performance_metrics_by_genre.items():
            benchmark_r2 = perf_dict['r_squared']
            scoring_r2 = scoring_performance_metrics_by_genre[genre]['r_squared']
            if scoring_r2 != None and benchmark_r2 != None:
                if benchmark_r2 - scoring_r2 > 0.1:
                    genres_with_degraded_metric[genre] = {}
                    genres_with_degraded_metric[genre]['benchmark_r_squared'] = benchmark_r2
                    genres_with_degraded_metric[genre]['scoring_r_squared'] = scoring_r2
                else:
                    genres_with_stable_metric[genre] = {}
                    genres_with_stable_metric[genre]['benchmark_r_squared'] = benchmark_r2
                    genres_with_stable_metric[genre]['scoring_r_squared'] = scoring_r2

        degraded_metrics['By Genre'] = genres_with_degraded_metric
        stable_metrics['By Genre'] = genres_with_stable_metric

        return degraded_metrics, stable_metrics

    @staticmethod
    def _get_metric_comparison_outcomes(
        champion_performance_metrics: Dict[str, Any],
        challenger_performance_metrics: Dict[str, Any],
        relevant_metric_names: List[str],
        degraded_or_stable: str,
        tolerance: float = 0.05
    ) -> Dict[str, Any]:
        """
        Compare performance metrics between the champion and challenger models.

        Args:
            champion_performance_metrics: Metrics of the current production model.
            challenger_performance_metrics: Metrics of the newly trained model.
            relevant_metric_names: List of metrics to compare.
            degraded_or_stable: Whether the metrics are considered degraded or stable.
            tolerance: Threshold for acceptable performance difference.

        Returns:
            Dictionary summarizing the comparison outcomes for each metric.
        """
        metric_comparison_outcomes = {}

        for metric_name in relevant_metric_names:
            if metric_name == 'Overall':
                champion_r2 = champion_performance_metrics['Overall']['r_squared']
                challenger_r2 = challenger_performance_metrics['Overall']['r_squared']
                if None not in [champion_r2, challenger_r2]:
                    if degraded_or_stable == 'degraded':
                        if challenger_r2 > champion_r2:
                            comparison_outcome = 'improvement'
                        else:
                            delta = challenger_r2 - champion_r2
                            comparison_outcome = 'no improvement' if delta >= -tolerance else 'degradation'
                    elif degraded_or_stable == 'stable':
                        delta = challenger_r2 - champion_r2
                        comparison_outcome = 'remained stable' if delta >= -tolerance else 'degradation'
                    metric_comparison_outcomes['Overall'] = comparison_outcome
                else:
                    logger_trainer.info(f"'Overall' below sample size threshold for performance evaluation, therefore will not be considered for comparison (as not considered a 'Critical' metric).") # unlikely condition...
            else:
                metric_comparison_outcomes.setdefault('By Genre', {})
                champion_r2 = champion_performance_metrics['By Genre'][metric_name]['r_squared']
                challenger_r2 = challenger_performance_metrics['By Genre'][metric_name]['r_squared']
                if None not in [champion_r2, challenger_r2]:
                    if degraded_or_stable == 'degraded':
                        if challenger_r2 > champion_r2:
                            comparison_outcome = 'improvement'
                        else:
                            delta = challenger_r2 - champion_r2
                            comparison_outcome = 'no improvement' if delta >= -tolerance else 'degradation'
                    elif degraded_or_stable == 'stable':
                        delta = challenger_r2 - champion_r2
                        comparison_outcome = 'remained stable' if delta >= -tolerance else 'degradation'
                    metric_comparison_outcomes['By Genre'][metric_name] = comparison_outcome
                else:
                    logger_trainer.info(f"'{metric_name}' segment below sample size threshold for performance evaluation, therefore will not be considered for comparison (as not considered a 'Critical' metric).") # the definition here for a 'Critical' metric is one where the sample size is above the minimum threshold (set to 100) during retraining performance evaluation as opposed to initial scoring performance evaluation, as the former will represent the smaller of the two datasets, seeing as evaluation is carried out on just the holdout test set, as opposed to the whole dataset from the time period. 

        return metric_comparison_outcomes

    @staticmethod
    def _get_high_level_metric_comparison_outcomes(
        degraded_metric_comparison_outcomes: Dict[str, Any],
        stable_metric_comparison_outcomes: Dict[str, Any]
    ) -> Tuple[bool, bool, bool]:
        """
        Aggregate and summarize comparison outcomes across all degraded and stable metrics.

        Args:
            degraded_metric_comparison_outcomes: Comparison outcomes for degraded metrics.
            stable_metric_comparison_outcomes: Comparison outcomes for stable metrics.

        Returns:
            Tuple of booleans:
                - any_degraded_improved
                - any_degraded_further_degraded
                - any_stable_degraded
        """
        degraded_outcomes_listed = []
        stable_outcomes_listed = []

        for outcomes_dict in [degraded_metric_comparison_outcomes, stable_metric_comparison_outcomes]:

            overall_outcome = outcomes_dict.get('Overall')

            if outcomes_dict == degraded_metric_comparison_outcomes:
                degraded_outcomes_listed.append(overall_outcome)
            elif outcomes_dict == stable_metric_comparison_outcomes:
                stable_outcomes_listed.append(overall_outcome)

            by_genre = outcomes_dict.get('By Genre', {})

            if outcomes_dict == degraded_metric_comparison_outcomes:
                degraded_outcomes_listed.extend(by_genre.values())
            elif outcomes_dict == stable_metric_comparison_outcomes:
                stable_outcomes_listed.extend(by_genre.values())

        any_degraded_improved = any(outcome == 'improvement' for outcome in degraded_outcomes_listed)
        any_degraded_further_degraded = any(outcome == 'degradation' for outcome in degraded_outcomes_listed)
        any_stable_degraded = any(outcome == 'degradation' for outcome in stable_outcomes_listed)

        return any_degraded_improved, any_degraded_further_degraded, any_stable_degraded

    def _get_overall_retraining_outcome(
        self,
        degraded_metric_comparison_outcomes: Dict[str, Any],
        stable_metric_comparison_outcomes: Dict[str, Any]
    ) -> str:
        """
        Determine whether to promote, reject, or flag the challenger model for human review.

        Args:
            degraded_metric_comparison_outcomes: Comparison outcomes for degraded metrics.
            stable_metric_comparison_outcomes: Comparison outcomes for stable metrics.

        Returns:
            String decision outcome ('Promote Challenger', 'Reject Challenger', or 'Human Review').
        """
        any_degraded_improved, any_degraded_further_degraded, any_stable_degraded = self._get_high_level_metric_comparison_outcomes(degraded_metric_comparison_outcomes, stable_metric_comparison_outcomes)

        decision_matrix = {
            (False, False, False): "Reject Challenger",
            (True,  False, False): "Promote Challenger",
            (False, True,  False): "Reject Challenger",
            (False, False, True):  "Reject Challenger",
            (True,  True,  False): "Human Review",
            (False, True,  True):  "Reject Challenger",
            (True,  False, True):  "Human Review",
            (True,  True,  True):  "Human Review",
        }

        overall_outcome = decision_matrix.get((any_degraded_improved, any_degraded_further_degraded, any_stable_degraded), "N/A")
        
        return overall_outcome

    @staticmethod
    def _notify_team(
        subject: str,
        message: str,
        recipients: Union[str, List[str]],
        sender_email: str,
        sender_password: str,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587
    ) -> None:
        """
        Send an email notification to the team when human review is required.

        Args:
            subject: Email subject line.
            message: Email body content.
            recipients: Single recipient email or list of emails.
            sender_email: Email address used to send the message.
            sender_password: Password or app password for the sender email.
            smtp_server: SMTP server address.
            smtp_port: SMTP server port.
        """
        if isinstance(recipients, str):
            recipients = [recipients]

        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        msg.attach(MIMEText(message, "plain"))

        try:
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            logger_trainer.info(f"Notification sent to {', '.join(recipients)}.")
        except Exception:
            logger_trainer.info(f"Failed to send notification.")
            raise