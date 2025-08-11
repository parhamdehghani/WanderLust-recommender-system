# WanderLust: A Hybrid Hotel Recommender System

This repository contains the end-to-end development and deployment of "WanderLust," a sophisticated hybrid hotel recommender system based on [Datafiniti hotel review dataset](https://www.kaggle.com/datasets/datafiniti/hotel-reviews) (confined to US hotels). The project leverages a combination of deep learning for semantic understanding and collaborative filtering for personalization, all wrapped in a scalable, serverless API deployed on Google Cloud.

## Features
- **Hybrid Recommendation:** Combines content-based semantic search with collaborative filtering to provide robust and personalized recommendations.
- **Deep Learning Semantic Search:** Utilizes a `Sentence-Transformer` model, fine-tuned with Triplet Loss on hotel review data, to understand the nuances of user queries. Fine-tuning data are created 50/50 with the idea of getting same-hotel reviews and good reviews close together (created separately and combined together).
- **Personalized Re-ranking:** Employs boosted decision trees to re-rank semantically relevant hotels based on a user's historical taste profile, captured by SVD.
- **Location Filtering:** Allows users to pre-filter recommendations by city and country for geographically relevant results.
- **Scalable Serverless API:** The final application is containerized with Docker and deployed on Google Cloud Run.

## Recommender System Design Concept

This system is built using a two-stage "candidate generation -> re-ranking" architecture. This is a common and highly effective pattern used in large-scale, production-grade recommender systems.

### Stage 1: Candidate Generation (The Broad Search)
The goal of this first stage is to quickly and efficiently sift through the entire catalog of thousands of hotels and select a smaller, more manageable set of a few hundred potentially relevant candidates.

**Method:** This stage is powered by our fast, content-based semantic search. It uses the fine-tuned sentence-transformer model to find the hotels whose content is most semantically similar to the user's query.


### Stage 2: Re-ranking (The Personalized Recommendations
The goal of this second stage is to take the smaller set of relevant candidates and apply a more powerful, computationally expensive model to create a final, precise, and personalized ranking.

**Method:** This stage is powered by our trained XGBoost model. For each of the candidate hotels, it creates a rich feature vector (combining user SVD factors, hotel SVD factors, and hotel content embeddings) and predicts a final, nuanced score.


This two-stage approach is both **scalable** (the expensive model only runs on a small subset of items) and **effective** (it combines a broad semantic search with a deep, personalized final ranking).

## Project Architecture

The project is divided into two main phases: an offline training phase where all the heavy models and assets are created, and an online inference phase where these assets are used to serve live recommendations.

### Offline Training Phase
This phase involves a series of Jupyter notebooks that perform the following steps:
1.  **Data Processing:** Loading, cleaning, and consolidating hotel review data.
2.  **Model Fine-Tuning:** Fine-tuning a `Sentence-Transformer` model (all-mpnet-base-v2) on review triplets to create a domain-specific text embedding model.
3.  **Feature & Asset Generation:**
    - **Content Embeddings:** Using the fine-tuned model to generate a content embedding for each hotel.
    - **Collaborative Factors:** Using `TruncatedSVD` on a user-item rating matrix to generate user and item factor vectors.
4.  **Ranker Training:** Training an `XGBoost` model on a combination of content and collaborative features to act as a final, predictive ranker.

### Online Inference & Deployment Architecture
The online component is a FastAPI application that serves the recommendations.

1.  A user sends a request with a query and optional (`user_id`, `city`, `country`) to the Cloud Run API endpoint.
2.  The application first pre-filters candidates based on `city` and `country`.
3.  It then uses the fine-tuned language model to perform a semantic search over the candidate content embeddings, generating a list of the top 100 most relevant hotels (**Candidate Generation**).
4.  If a `user_id` is provided, the XGBoost ranker uses the user's SVD profile to re-rank these 100 candidates for personalization (**Re-ranking**).
5.  The final top 5 hotels are returned to the user as a JSON response.

## Technology Stack
- **Cloud Platform:** Google Cloud (Vertex AI Workbench, GCS, Artifact Registry, Cloud Run)
- **ML / Data Science:**
  - **Deep Learning:** PyTorch, Sentence-Transformers
  - **Classical ML:** Scikit-learn (SVD), XGBoost
  - **Data Handling:** Pandas, NumPy
- **Deployment:**
  - **API Framework:** FastAPI
  - **Web Server:** Uvicorn
  - **Containerization:** Docker

## Repository Structure
```
.
├── 01_data_preparation.ipynb       # Notebook for initial data loading and cleaning
├── 02_model_finetuning.ipynb       # Notebook for fine-tuning the sentence-transformer
├── 03_feature_extraction.ipynb     # Notebook for generating content & SVD features
├── 04_ranker_model.ipynb           # Notebook for training the XGBoost ranker
├── 05_inference_function.ipynb     # Notebook for testing the final function and launching the Gradio demo
├── main.py                         # FastAPI application script for deployment
├── Dockerfile                      # Docker instructions for building the container
└── requirements.txt                # Python dependencies for the project
```

## Usage

### Running the Notebooks
The notebooks are designed to be run sequentially in a Google Vertex AI Workbench environment. Each notebook saves its output (data files, models) to a Google Cloud Storage bucket, which is then used as the input for the next notebook in the sequence.

### Interacting with the Deployed API
Recommender service is already deployed on Google Cloud Run employing the containarized ML pipeline, and anyone can interact with it by sending a request (via `curl` or browser). The deployed API endpint is using the developed collaborative and content-based features as well as the fine-tuned model to personalize recommendations.

Deployed API (may take some time to load data for the first time as the service is serverless): https://recommender-service-86763462033.northamerica-northeast1.run.app/

**Example 1: Anonymous, Location-Specific Query**
```bash
curl "https://recommender-service-86763462033.northamerica-northeast1.run.app/recommend?query=a%20modern%20hotel%20in%20Los%20Angeles&city=Los%20Angeles"
```

**Example 2: Personalized Query**
```bash
curl "https://recommender-service-86763462033.northamerica-northeast1.run.app/recommend?query=a%20stylish%20place%20with%20a%20great%20view&user_id=50&city=Vancouver"
```

## Key Concepts Demonstrated
- **Hybrid Recommender Systems:** Combining content-based and collaborative filtering methods.
- **Deep Learning for NLP:** Fine-tuning transformer models (encoder-only) for domain-specific semantic understanding using generated triplets as the training set and Triplet Loss.  
- **Matrix Factorization:** Using SVD to uncover latent features in user-item interaction data.
- **Production ML Architecture:** Implementing a two-stage "candidate generation -> re-ranking" pattern, which is standard in large-scale recommender systems.
- **Serverless ML Deployment:** Containerizing a full ML stack with Docker and deploying it on a scalable, serverless platform (Google Cloud Run).
