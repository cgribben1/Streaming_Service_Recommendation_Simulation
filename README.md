# Film Recommender Engine

> [!IMPORTANT]
> **It is highly recommended that you refer to the two below resources in order to get the best possible sense of the project!**  
> - **Full technical write-up:** `docs/overview.docx`  
> - **End-to-end demonstration notebook:** `examples/demo.ipynb`

This repository represents a **film recommendation engine** built using collaborative filtering (matrix factorisation), and implemented within a full end-to-end simulation framework for evaluating model performance, monitoring key prediction metrics, detecting data distribution drift, and executing retraining strategies.

The engine is structured as a Python package (`film_recommender_cg`) that would conceptually be adapted for deployment across regions and markets. The PoC demonstrates modular architecture, offline batch scoring, model lifecycle management, and monitoring strategies aligned with real‑world MLOps practices.

---

## Project Overview

This project implements a modular, production-inspired recommendation engine packaged as:

`film_recommender_cg`

It demonstrates:

- Collaborative filtering using **`scikit-surprise`** SVD (Singular Value Decomposition) model
- Cold-start handling through popularity-based fallback recommendations
- Batch scoring, monitoring, and model lifecycle management
- Simulation of user behaviour under shifting data distributions
- Drift detection and challenger-vs-champion retraining logic
- Usage of real MLOps tools (Weights & Biases, Databricks, MongoDB)

The design mirrors how recommendation engines operate in commercial environments with fixed content catalogs, evolving user cohorts, and regular retraining cycles.

---

## Core Methodology

### Collaborative Filtering  
Users with at least five interactions are served by a matrix-factorisation model. True cold-start users are routed to a popularity-based fallback.

### Dataset  
The system uses a subset of **The Movies Dataset** (Kaggle).  
To avoid memory issues, the model is trained on the first **120,000 rows**, covering ~1.8k users and ~4k films.

### Simulation Engine  
A chronological simulation loop generates synthetic user interactions:

- Each user consumes 1–20 films per round  
- The model scores all unseen items  
- Users select films using weighted sampling (favours high-ranked items)  
- Ratings = predicted score + calibrated noise  
- Already-seen films are masked at inference  

A calibrated noise factor ensures simulated R² resembles real-world performance instead of inflating metrics.

---

## Monitoring & Evaluation

The system tracks several key recommender-quality dimensions:

- **Accuracy** – R² on held-out/synthetic data  
- **Coverage** – proportion of catalog ever recommended  
- **Diversity** – average cosine distance between recommended items  
- **Personalization** – how different recommendations are across users  

Per-genre segmentation is used for drift detection, ignoring segments with <100 interactions to reduce noise.

The simulation allows “what-if” drift experiments, such as penalising specific genres to trigger retraining.

---

## Retraining Logic

Retraining is triggered when:

- Global R² drops below the established baseline, or  
- Any genre-level metric deteriorates

Challenger models are **only promoted** if:

- They improve degraded metrics  
- They do **not** harm stable segments  
- No new regressions are introduced  

Rejected challengers mirror the human-review workflow used in real production systems.

---

## Architecture & Tools

- **VSCode** – development and orchestration  
- **MongoDB** – ratings storage  
- **Weights & Biases** – experiment tracking and model evaluation  
- **Python package structure** – clean separation of trainer, scorer, and utilities  

Training uses historical interactions only.  
Inference requires the full ratings table to correctly filter previously watched films.

---

## Installation

The `film_recommender_cg` package is available on [PyPI](https://pypi.org/) and can be installed using `pip`:

```bash
pip install film_recommender_cg
```

---

## Future Directions

Potential next steps would include:

- Neural recommenders (two-tower, embedding-based, deep MF)
- Hybrid models with metadata (genres, languages, keywords, embeddings)
- Time-aware or rolling-window training
- Real-time inference APIs
- Exploration-vs-exploitation algorithms to improve coverage and diversity

---

## Full Documentation

Once again, a complete technical overview—including methodology, architecture, simulation design, and retraining logic—is available in:

**`docs/overview.docx`**

---

## License

Apache 2.0 License

---

 
