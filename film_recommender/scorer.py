'''
scorer.py
'''

import os
import json
import pickle
import logging
import random
import gc
from itertools import chain, product
import pandas as pd
import numpy as np
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression

logger_scorer = logging.getLogger('scorer') # TODO: change back to __name__

class Scorer:

    def __init__(self, sql_connector, wandb_auth_key, wandb_project_path, wandb_project_name, new_user_threshold, n_recs, max_interactions_between_scorings):
        self.sql_connector = sql_connector
        self.new_user_threshold = new_user_threshold
        self.wandb_auth_key = wandb_auth_key
        self.wandb_project_path = wandb_project_path
        self.wandb_project_name = wandb_project_name
        self.n_recs = n_recs
        self.max_interactions_between_scorings = max_interactions_between_scorings
        self.wandb_run = None
        self.linked_model_artifact = None
        self.ratings = None
        self.rating_counts = None
        self.ratings_old = None
        self.ratings_new = None
        self.all_users = None
        self.old_users = None
        self.new_users = None
        self.genres_table = None
        self.model = None
        self.rankings_generated_old_users = None
        self.rankings_generated_new_users = None
        self.rankings_generated_all_users = None
        self.new_interactions_old_users = None
        self.new_interactions_new_users = None
        self.new_interactions_all_users = None
        self.genre_avg_count_in_top_n_recs_per_user_group = None
        self.coverage_per_user_group = None
        self.avg_diversity_score_per_user_group = None
        self.avg_personalization_score_per_user_group = None
        self.performance_metrics_per_user_group = None

        wandb.login(key=wandb_auth_key)
        self.wandb_run = wandb.init(project=wandb_project_name, job_type='scoring', tags=["scoring"])
        artifact_path = os.path.join(wandb_project_path, 'svd_model:prod')
        self.linked_model_artifact = self.wandb_run.use_artifact(artifact_path, type="model")

        logger_scorer.info("'Scorer' object instantiated; W&B run initialized.")

    def close_wandb_run(self):
        if self.wandb_run:
            self.wandb_run.finish()
            logger_scorer.info("Scoring W&B run closed.")
        else:
            logger_scorer.info("No scoring W&B run currently open, therefore nothing to close.")

    def score_simulate_and_update_ratings_table(self, popularity_penalty_coef=0, popularity_transformation_for_penalty='Normalization', genre_to_penalize=None, genre_penalty=None):
        global ratings ### temp

        try:
            try:
                ratings, rating_counts, ratings_old, ratings_new, all_users, old_users, new_users = self._load_ratings_data()
                logger_scorer.info("Ratings data loaded successfully.")
            except Exception:
                logger_scorer.exception("Failed to load ratings data.")
                raise

            try:
                genres_table = self._load_genres_table()
                logger_scorer.info("Genres table loaded successfully.")
            except Exception:
                logger_scorer.exception("Failed to load genres table.")
                raise

            try:
                model = self._import_model()
                logger_scorer.info("Model imported successfully.")
            except Exception:
                logger_scorer.exception("Failed to import model.")
                raise

            #self.rankings_generated_all_users = None is this needed ?

            try:
                rankings_generated_old_users = self._get_ranked_films_per_user_old(ratings_old, old_users, model, ratings, popularity_penalty_coef, popularity_transformation_for_penalty)
                logger_scorer.info("Rankings for old users generated successfully.")
            except Exception:
                logger_scorer.exception("Failed to generate rankings for old users.")
                raise

            try:
                rankings_generated_new_users = self._get_ranked_films_per_user_new(ratings, ratings_new, new_users)
                logger_scorer.info("Rankings for new users generated successfully.")
            except Exception:
                logger_scorer.exception("Failed to generate rankings for new users.")
                raise

            try:
                new_interactions_old_users = self._simulate_batch_of_film_choices(old_users, rankings_generated_old_users, ratings,genres_table, genre_to_penalize, genre_penalty, 'old')
                logger_scorer.info("New interactions for old users simulated successfully.")
            except Exception:
                logger_scorer.exception("Failed to simulate film choices for old users.")
                raise

            try:
                new_interactions_new_users = self._simulate_batch_of_film_choices(new_users, rankings_generated_new_users, ratings,genres_table, genre_to_penalize, genre_penalty, 'new')
                logger_scorer.info("New interactions for new users simulated successfully.")
            except Exception:
                logger_scorer.exception("Failed to simulate film choices for new users.")
                raise

            try:
                new_interactions_all_users = self._update_ratings_table(new_interactions_old_users, new_interactions_new_users)
                logger_scorer.info("Ratings table updated successfully.")
            except Exception:
                logger_scorer.exception("Failed to update ratings table.")
                raise

            try:
                self.ratings = ratings
                self.rating_counts = rating_counts
                self.ratings_old = ratings_old
                self.ratings_new = ratings_new
                self.all_users = all_users
                self.old_users = old_users
                self.new_users = new_users
                self.genres_table = genres_table
                self.model = model
                self.rankings_generated_old_users = rankings_generated_old_users
                self.rankings_generated_new_users = rankings_generated_new_users
                self.rankings_generated_all_users = pd.concat([rankings_generated_old_users, rankings_generated_new_users], axis=0)
                self.new_interactions_old_users = new_interactions_old_users
                self.new_interactions_new_users = new_interactions_new_users
                self.new_interactions_all_users = new_interactions_all_users
            except Exception:
                logger_scorer.exception("Failed to assign attributes to Scorer object following running of 'score_simulate_and_update_ratings_table()'.")
                raise

            logger_scorer.info("All processing completed successfully.")

            try:
                del rankings_generated_old_users
                del rankings_generated_new_users
                del new_interactions_old_users
                del new_interactions_new_users
                del new_interactions_all_users
            except Exception:
                pass
            gc.collect()
            logger_scorer.debug("Garbage collection complete.")

        except Exception:
            logger_scorer.exception("Error occurred in running 'score_simulate_and_update_ratings_table()'.")
            self.close_wandb_run()
            raise
        
    def get_genre_avg_counts_in_top_n_recs(self):
        try:
            genre_avg_count_in_top_n_recs_per_user_group = {}
            for user_group in ['all', 'new', 'old']:
                genre_avg_count_in_top_n_recs = self._get_genre_avg_counts_in_top_n_recs(user_group)
                genre_avg_count_in_top_n_recs_per_user_group[f"{user_group}_users"] = genre_avg_count_in_top_n_recs

            self.genre_avg_count_in_top_n_recs_per_user_group = genre_avg_count_in_top_n_recs_per_user_group
            return genre_avg_count_in_top_n_recs_per_user_group

        except Exception:
            logger_scorer.exception("Error occurred in getting genre average counts in top 'n' recommendations metrics.")
            self.close_wandb_run()
            raise

    def _get_genre_avg_counts_in_top_n_recs(self, user_group):
        len_each_ranking_generated = self.n_recs + self.max_interactions_between_scorings
        top_n_to_analyse = 100 if len_each_ranking_generated > 100 else len_each_ranking_generated

        if user_group == 'all':
            rankings_generated = self.rankings_generated_all_users
        elif user_group == 'old':
            rankings_generated = self.rankings_generated_old_users
        elif user_group == 'new':
            rankings_generated = self.rankings_generated_new_users

        user_ids = rankings_generated.index
        genres_table = self.genres_table
        unique_genres = list(set(chain.from_iterable(genres_table['genres'].values)))

        genre_counts = {i: 0 for i in unique_genres}
        for user_id in user_ids:
            single_ranking_dict = rankings_generated[user_id]
            single_ranking_df = pd.DataFrame(list(single_ranking_dict.items()), columns=['film_id', 'predicted_rating'])
            single_ranking_df = single_ranking_df.merge(genres_table, how='left', on=['film_id'])[:top_n_to_analyse]

            for genre in unique_genres:
                flattened_recommended_genres_column = chain.from_iterable(single_ranking_df['genres'].values)
                genre_count = sum(1 for i in flattened_recommended_genres_column if i == genre)
                genre_counts[genre] += genre_count

        genre_avg_count_in_top_n_recs = {k: round((v / len(user_ids)), 1) for k, v in genre_counts.items()}

        logger_scorer.info(f"### Genre average counts in top {top_n_to_analyse} recommendations ({user_group} users): ###")
        for genre, count in genre_avg_count_in_top_n_recs.items():
            logger_scorer.info(f"{genre} = {count}")

        self._log_artifact(f"scoring_genre_avg_count_in_top_n_recs_{user_group}_users", genre_avg_count_in_top_n_recs)

        return genre_avg_count_in_top_n_recs

    def get_coverage_scores(self):
        try:
            coverage_score_per_user_group = {}
            for user_group in ['all', 'new', 'old']:
                coverage_score = self._get_coverage_score(user_group)
                coverage_score_per_user_group[f"{user_group}_users"] = coverage_score

            self.coverage_score_per_user_group = coverage_score_per_user_group
            return coverage_score_per_user_group

        except Exception:
            logger_scorer.exception("Error occurred in getting coverage metrics.")
            self.close_wandb_run()
            raise

    def _get_coverage_score(self, user_group):
        if user_group == 'all':
            rankings_generated = self.rankings_generated_all_users
        elif user_group == 'old':
            rankings_generated = self.rankings_generated_old_users
        elif user_group == 'new':
            rankings_generated = self.rankings_generated_new_users

        unique_recommended_films = set()
        for dct in rankings_generated:
            unique_recommended_films.update(dct.keys())

        number_of_unique_film_recs = len(unique_recommended_films)
        total_catalogue_size = len(self.ratings['film_id'].unique())
        coverage = number_of_unique_film_recs / total_catalogue_size

        logger_scorer.info(f"### Coverage for scoring session ('{user_group.title()}' users): {coverage * 100:.2f}% ###")

        coverage_dict = {'coverage': coverage}
        self._log_artifact(f"scoring_coverage_{user_group}_users", coverage_dict)

        return coverage

    def get_avg_diversity_scores(self):
        try:
            avg_diversity_score_per_user_group = {}
            for user_group in ['all', 'new', 'old']:
                avg_score = self._get_avg_diversity_score(user_group)
                avg_diversity_score_per_user_group[f"{user_group}_users"] = avg_score

            self.avg_diversity_scores_per_user_group = avg_diversity_score_per_user_group
            return avg_diversity_score_per_user_group

        except Exception:
            logger_scorer.exception("Error occurred in getting average diversity scores.")
            self.close_wandb_run()
            raise

    def _get_avg_diversity_score(self, user_group):
        if user_group == 'all':
            rankings_generated = self.rankings_generated_all_users
        elif user_group == 'old':
            rankings_generated = self.rankings_generated_old_users
        elif user_group == 'new':
            rankings_generated = self.rankings_generated_new_users

        model = self.model
        film_embeddings = {model.trainset.to_raw_iid(i): model.qi[i] for i in range(len(model.qi))}

        user_diversity_scores = []
        for user_id, film_dict in rankings_generated.items():
            film_ids = list(film_dict.keys())
            vectors = [film_embeddings[film_id] for film_id in film_ids if film_id in film_embeddings]
            vectors = np.array(vectors)
            sim_matrix = cosine_similarity(vectors)
            n = len(vectors)
            pairwise_sims = sim_matrix[np.triu_indices(n, k=1)]
            pairwise_dists = 1 - pairwise_sims
            avg_distance = pairwise_dists.mean()
            user_diversity_scores.append(avg_distance)

        avg_diversity_score = sum(user_diversity_scores) / len(user_diversity_scores)
        logger_scorer.info(f"### Mean diversity score for scoring session ('{user_group.title()}'): {avg_diversity_score:.2f} ###")

        plt.boxplot(user_diversity_scores)
        plt.title(f"User Diversity Scores ('{user_group.title()}')")
        plt.ylabel("Diversity Score")
        plt.show()

        avg_diversity_score_dict = {'mean_diversity_score': avg_diversity_score}

        self._log_artifact(f"scoring_avg_diversity_score_{user_group}_users", avg_diversity_score_dict)

        return avg_diversity_score

    def get_avg_personalization_scores(self, plot_scores_by_num_films_rated=False):
        try:
            personalization_scores_per_user_group = {}
            for user_group in ['all', 'new', 'old']:
                score = self._get_avg_personalization_score(user_group, plot_scores_by_num_films_rated)
                personalization_scores_per_user_group[f"{user_group}_users"] = score

            self.avg_personalization_scores_per_user_group = personalization_scores_per_user_group
            return personalization_scores_per_user_group

        except Exception:
            logger_scorer.exception("Error occurred in getting average personalization scores.")
            self.close_wandb_run()
            raise

    def _get_avg_personalization_score(self, user_group, plot_scores_by_num_films_rated=False):
        if user_group == 'all':
            rankings_generated = self.rankings_generated_all_users
        elif user_group == 'old':
            rankings_generated = self.rankings_generated_old_users
        elif user_group == 'new':
            rankings_generated = self.rankings_generated_new_users

        all_film_ids = sorted({film_id for recs in rankings_generated.values() for film_id in recs.keys()})
        film_index_map = {film_id: idx for idx, film_id in enumerate(all_film_ids)}

        user_vectors = []
        user_ids = []
        for user_id, film_dict in rankings_generated.items():
            vec = np.zeros(len(all_film_ids))
            for film_id, pred_rating in film_dict.items():
                vec[film_index_map[film_id]] = pred_rating
            user_vectors.append(vec)
            user_ids.append(user_id)

        all_user_recs_matrix = np.array(user_vectors)
        cosine_similarities_matrix = cosine_similarity(all_user_recs_matrix)
        n = len(user_ids)
        cosine_similarities_list = cosine_similarities_matrix[np.triu_indices(n, k=1)]

        avg_similarity = cosine_similarities_list.mean()
        avg_personalization_score = 1 - avg_similarity

        logger_scorer.info(f"### Average personalization score ('{user_group.title()}'): {avg_personalization_score:.3f} ###")

        plt.boxplot(cosine_similarities_list)
        plt.title(f"User Personalization Scores ('{user_group.title()}')")
        plt.ylabel("Personalization Score")
        plt.show()

        if plot_scores_by_num_films_rated:
            rating_counts_df = self.rating_counts
            per_user_avg_similarities = (cosine_similarities_matrix.sum(axis=1) - 1) / (n - 1)
            per_user_personalization = 1 - per_user_avg_similarities
            personalization_scores_df = pd.DataFrame({"user_id": user_ids, "personalization_score": per_user_personalization})
            merged_df = rating_counts_df.merge(personalization_scores_df, on="user_id")
            X, y = merged_df[["num_rated_films"]], merged_df["personalization_score"]

            plt.scatter(X, y, alpha=0.6)
            plt.xlabel("Number of Films Rated")
            plt.ylabel("Personalization Score")
            plt.title("Personalization vs. Number of Films Rated")
            plt.show()

            r_squared = LinearRegression().fit(X, y).score(X, y)
            corr = np.corrcoef(X.squeeze(), y)[0, 1]
            logger_scorer.info(f"R² for Personalization vs. Films Rated: {r_squared:.3f}")
            logger_scorer.info(f"Pearson correlation for Personalization vs. Films Rated: {corr:.3f}")

        avg_personalization_score_dict = {'avg_personalization_score': avg_personalization_score}

        self._log_artifact(f"scoring_avg_personalization_score_{user_group}_users", avg_personalization_score_dict)

        return avg_personalization_score

    def evaluate_performance_metrics(self):
        try:
            performance_metrics_per_user_group = {}
            for user_group in ['all', 'new', 'old']:
                metrics = self._evaluate_performance_metrics(user_group)
                performance_metrics_per_user_group[f"{user_group}_users"] = metrics

            self.performance_metrics_per_user_group = performance_metrics_per_user_group
            return performance_metrics_per_user_group

        except Exception:
            logger_scorer.exception("Error occurred in evaluating scoring performance metrics.")
            self.close_wandb_run()
            raise

    def _evaluate_performance_metrics(self, user_group):
        logger_scorer.info("Evaluating model performance on new (simulated) data...")

        all_performance_metrics = {
            'Overall': self._evaluate_performance_metrics_overall(user_group),
            'By Genre': self._evaluate_performance_metrics_by_genre(user_group)
        }

        self._log_artifact(f"scoring_performance_metrics_{user_group}_users", all_performance_metrics)
        logger_scorer.info("Evaluation complete!")
        return all_performance_metrics

    def _load_ratings_data(self):
        logger_scorer.info("Loading in 'ratings' data...")

        #ratings = self.sql_connector.read_sql_table('ratings', sort_on='user_id').drop('created_at', axis=1)[:120000]
        ### ratings is currently global...

        rating_counts = ratings.groupby('user_id').size().reset_index(name='num_rated_films')
        ratings_with_counts = ratings.merge(rating_counts, on="user_id")

        ratings_old = ratings_with_counts[ratings_with_counts["num_rated_films"] > self.new_user_threshold].drop('num_rated_films', axis=1).reset_index(drop=True)
        ratings_new = ratings_with_counts[ratings_with_counts["num_rated_films"] <= self.new_user_threshold].drop('num_rated_films', axis=1).reset_index(drop=True)

        all_users = list(range(1, ratings['user_id'].max() + 1))
        old_users = sorted(ratings_old['user_id'].unique())
        new_users = list(set(all_users) - set(old_users))

        logger_scorer.info("Data successfully loaded!")
        logger_scorer.info(f"Number of 'Old' users: {len(old_users)} -> Using SVD collaborative filtering model to generate these recommendations...")
        logger_scorer.info(f"Number of 'New' users: {len(new_users)} -> Falling back to cold-start strategy of recommending top 'n' films by popularity...")

        return ratings, rating_counts, ratings_old, ratings_new, all_users, old_users, new_users
    
    @staticmethod
    def _simplify_genres_list(x):
        raw_genres_list = eval(x)
        simplified_genres_list = [raw_genres_list[i]['name'] for i in range(len(raw_genres_list))]
        
        return simplified_genres_list
    
    @staticmethod
    def _fill_in_missing_ids(genres_table):
        full_film_ids = pd.Series(range(1, 5000), name='film_id')
        missing_film_ids = full_film_ids[~full_film_ids.isin(genres_table['film_id'])]

        missing_rows = pd.DataFrame({
            'film_id': missing_film_ids,
            'genres': [['Undisclosed']] * len(missing_film_ids)
        })

        genres_table = pd.concat([genres_table, missing_rows], ignore_index=True)
        genres_table = genres_table.sort_values('film_id').reset_index(drop=True)

        return genres_table

    def _load_genres_table(self):
        #genres_table = self.sql_connector.read_sql_table('titles_and_genres')[['film_id', 'genres']]
        genres_table = pd.read_csv("../data/genres_table.csv")
        
        genres_table['genres'] = genres_table['genres'].apply(lambda x: self._simplify_genres_list(x))

        genres_table = self._fill_in_missing_ids(genres_table)

        return genres_table 

    def _import_model(self):
        logger_scorer.info("Loading in model from W&B...")

        model_dir = self.linked_model_artifact.download()
        model_file = os.path.join(model_dir, "svd_model.pkl")

        with open(model_file, "rb") as f:
            model = pickle.load(f)

        logger_scorer.info("Model successfully loaded!")

        return model

    def _log_artifact(self, label, data):
        tmp_path = f"{label}.json"  

        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=4)

        artifact = wandb.Artifact(label, type="metric")
        artifact.add_file(tmp_path)

        self.wandb_run.log_artifact(artifact)

        os.remove(tmp_path)

        logger_scorer.info(f"'{label}' successfully logged as artifact in W&B.")

    def _reconstruct_interaction_matrix_and_predict_ratings(self, ratings_old, old_users, model):
        all_films = sorted(ratings_old['film_id'].unique())
        full_index = list(product(old_users, all_films))
        predictions = [model.predict(user, film).est for user, film in full_index]

        df = pd.DataFrame(full_index, columns=['user_id', 'film_id'])
        df['predicted_rating'] = predictions

        past_interactions = ratings_old[["user_id", "film_id"]].copy()
        past_interactions["already_rated"] = True

        processed_chunks = []
        for batch in np.array_split(df, 10):
            merged = batch.merge(past_interactions, on=["user_id", "film_id"], how="left")
            merged.loc[merged["already_rated"] == True, "predicted_rating"] = 0
            merged.drop(columns=["already_rated"], inplace=True)
            processed_chunks.append(merged)

        full_matrix_df = pd.concat(processed_chunks, ignore_index=True)
        interaction_matrix = full_matrix_df.pivot(index='user_id', columns='film_id', values='predicted_rating')

        return interaction_matrix
    
    def _get_film_popularities(self, ratings, popularity_transformation_for_penalty):
        film_popularities = ratings.groupby('film_id')['rating'].count()

        if popularity_transformation_for_penalty == 'Normalization':
            min_count = film_popularities.min()
            max_count = film_popularities.max()

            film_popularities_normalized = ((film_popularities - min_count) / (max_count - min_count))

            return film_popularities_normalized.to_dict()

        elif popularity_transformation_for_penalty == 'Log-scaling':
            film_popularities_log_scaled =  np.log1p(film_popularities)
        
            min_count = film_popularities_log_scaled.min()
            max_count = film_popularities_log_scaled.max()
            
            film_popularities_log_scaled_and_normalized = ((film_popularities_log_scaled - min_count) / (max_count - min_count))

            return film_popularities_log_scaled_and_normalized.to_dict()

    def _penalize_films_by_popularity(self, user_pred_ratings_dict, film_popularities_dict, popularity_penalty_coef):
        for film_id, pred_rating in user_pred_ratings_dict.items():
            film_popularity = film_popularities_dict[film_id]

            penalized_pred_rating = max(0, pred_rating - popularity_penalty_coef * film_popularity)

            user_pred_ratings_dict[film_id] = penalized_pred_rating

        return user_pred_ratings_dict
    
    def _sort_and_crop_pred_ratings(self, user_pred_ratings_dict):
        return dict(sorted(user_pred_ratings_dict.items(), key=lambda x: x[1], reverse=True)[:self.n_recs + self.max_interactions_between_scorings])

    def _get_ranked_films_per_user_old(self, ratings_old, old_users, model, ratings, popularity_penalty_coef, popularity_transformation_for_penalty):
        logger_scorer.info("Scoring for 'old' users...")

        interaction_matrix = self._reconstruct_interaction_matrix_and_predict_ratings(ratings_old, old_users, model)

        pred_ratings_unpenalized = interaction_matrix.apply(lambda row: row.to_dict(), axis=1)

        del interaction_matrix
        gc.collect()

        film_popularities_dict = self._get_film_popularities(ratings, popularity_transformation_for_penalty)

        pred_ratings_popularity_penalized = pred_ratings_unpenalized.apply(lambda user_pred_ratings_dict: self._penalize_films_by_popularity(user_pred_ratings_dict, film_popularities_dict, popularity_penalty_coef))

        rankings_generated = pred_ratings_popularity_penalized.apply(lambda user_pred_ratings_dict: self._sort_and_crop_pred_ratings(user_pred_ratings_dict))

        logger_scorer.info("Scoring complete for 'old' users!")
        
        return rankings_generated

    @staticmethod
    def _sample_noisy_rating(chosen_film_genres, predicted_rating, genre_to_penalize, genre_penalty, calibration_factor=0.545):
        if genre_to_penalize and genre_penalty and chosen_film_genres:
                if genre_to_penalize.title() in chosen_film_genres:
                    predicted_rating *= genre_penalty

        possible_ratings = np.arange(0, 5.5, 0.5)
        distances = np.abs(possible_ratings - predicted_rating)
        probabilities = np.exp(-distances ** 2 / (2 * calibration_factor ** 2))
        probabilities /= probabilities.sum()
        noisy_rating = np.random.choice(possible_ratings, p=probabilities)

        return noisy_rating

    def _simulate_single_user_film_choice(self, user_id, ranking, n_recs, genres_table, genre_to_penalize, genre_penalty, decay=0.1):
        ranks = np.arange(n_recs)
        weights = np.exp(-decay * ranks)
        probabilities = weights / weights.sum()
        chosen_index = np.random.choice(n_recs, p=probabilities)
        chosen_film_id, predicted_rating = list(ranking.items())[chosen_index]
        chosen_film_genres = genres_table[genres_table['film_id'] == chosen_film_id]['genres'].values[0]

        simmed_rating = self._sample_noisy_rating(chosen_film_genres, predicted_rating, genre_to_penalize, genre_penalty)

        return [user_id, chosen_film_id, simmed_rating, predicted_rating, chosen_film_genres]

    def _simulate_batch_of_film_choices(self, user_segment, ranked_films_per_user, ratings, genres_table, genre_to_penalize, genre_penalty, user_group_label):
        logger_scorer.info(f"Simulating interactions for '{user_group_label}' users...")

        interactions = []
        for user_id in user_segment:
            n_recs = self.n_recs
            ranking = ranked_films_per_user[user_id].copy()
            for _ in range(random.randint(1, self.max_interactions_between_scorings)):
                interaction = self._simulate_single_user_film_choice(user_id, ranking, n_recs, genres_table, genre_to_penalize, genre_penalty)
                interactions.append(interaction)
                del ranking[interaction[1]]
                n_recs -= 1

        interactions_df = pd.DataFrame(interactions, columns=list(ratings.columns) + ['predicted_rating', 'genres'])

        logger_scorer.info(f"Simulation of '{user_group_label}' users complete!")

        return interactions_df

    def _get_ranked_films_per_user_new(self, ratings, ratings_new, new_users):
        logger_scorer.info("Scoring for 'new' users...")

        film_counts = ratings.groupby('film_id').size().reset_index(name='rating_count')
        film_means = ratings.groupby('film_id')['rating'].mean().reset_index(name='rating_mean')
        top_films = film_counts.merge(film_means, on='film_id')
        top_films['score'] = top_films['rating_count'] * top_films['rating_mean']
        top_films.sort_values(by='score', ascending=False, inplace=True)

        top_dict = top_films[['film_id', 'rating_mean']].head(self.n_recs + self.max_interactions_between_scorings).set_index('film_id')['rating_mean'].to_dict()

        films_per_user = ratings_new.groupby('user_id')['film_id'].apply(list).reset_index(name='rated_films')
        user_df = pd.DataFrame(new_users, columns=['user_id'])
        films_per_user = films_per_user.merge(user_df, on='user_id', how='right').fillna({'rated_films': '[]'})

        rankings_dict = {}
        for _, row in films_per_user.iterrows():
            rated = [] if row['rated_films'] == '[]' else row['rated_films']
            rankings_dict[row['user_id']] = {fid: score for fid, score in top_dict.items() if fid not in rated}

        rankings_generated = pd.Series(rankings_dict)
        rankings_generated.index.name = 'user_id'

        logger_scorer.info("Scoring complete for 'new' users!")

        return rankings_generated
    
    def _update_ratings_table(self, new_interactions_old_users, new_interactions_new_users):
        global ratings

        logger_scorer.info("Updating ratings table...")

        new_interactions_all_users = pd.concat([new_interactions_old_users, new_interactions_new_users], axis=0).reset_index(drop=True)
        new_interactions_all_users_subset_columns = new_interactions_all_users.drop(['predicted_rating', 'genres'], axis=1)

        ratings = pd.concat([ratings, new_interactions_all_users_subset_columns], axis=0) ### temp
        #self.sql_connector.append_df_to_sql_table(new_interactions_all_users_subset_columns, 'ratings')

        logger_scorer.info("Ratings table updated.")

        return new_interactions_all_users
    
    def _evaluate_performance_metrics_overall(self, user_group):
        if user_group == 'all':
            interactions = self.new_interactions_all_users
        elif user_group == 'old':
            interactions = self.new_interactions_old_users
        elif user_group == 'new':
            interactions = self.new_interactions_new_users

        predicted_ratings = interactions['predicted_rating'].values
        simmed_ratings = interactions['rating'].values
        
        rmse = round(np.sqrt(mean_squared_error(simmed_ratings, predicted_ratings)), 3)
        r2 = round(r2_score(simmed_ratings, predicted_ratings), 3)

        logger_scorer.info(f"### Following recent batch of simulated interactions, current model performance metrics ('{user_group.title()}' users): ###")
        logger_scorer.info(f"Overall rmse = {rmse}")
        logger_scorer.info(f"Overall r-squared = {r2}")

        performance_metrics = {'rmse': rmse, 'r_squared': r2}

        wandb.log(performance_metrics) # logging as metric as well as artifact (see 'evaluate_performance_metrics()' below)

        return performance_metrics
    
    def _evaluate_performance_metrics_by_genre(self, user_group):
        if user_group == 'all':
            interactions = self.new_interactions_all_users
        elif user_group == 'old':
            interactions = self.new_interactions_old_users
        elif user_group == 'new':
            interactions = self.new_interactions_new_users

        genres_table = self.genres_table

        unique_genres = list(set(chain.from_iterable(genres_table['genres'].values)))
        performance_metrics = {}

        logger_scorer.info(f"### Following recent batch of simulated interactions, current model performance metrics ('{user_group.title()}' users): ###")

        for genre in unique_genres:
            genre_relevant_interactions = interactions[interactions['genres'].apply(lambda x: genre in x)]

            if len(genre_relevant_interactions) > 100:
                predicted_ratings = genre_relevant_interactions['predicted_rating'].values
                simmed_ratings = genre_relevant_interactions['rating'].values
                
                rmse = round(np.sqrt(mean_squared_error(simmed_ratings, predicted_ratings)), 3)
                r2 = round(r2_score(simmed_ratings, predicted_ratings), 3)
            
                logger_scorer.info(f"'{genre}' rmse = {rmse}")
                logger_scorer.info(f"'{genre}' r-squared = {r2}")
            else:
                rmse = None
                r2 = None

                logger_scorer.info(f"'{genre}' rmse = Insufficient Data")
                logger_scorer.info(f"'{genre}' r-squared = Insufficient Data")

            performance_metrics[genre] = {'rmse': rmse, 'r_squared': r2}

        return performance_metrics
