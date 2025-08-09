import os
import subprocess
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from google.cloud import storage
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Constants and Global Variables ---
BUCKET_NAME = "wanderlust-recommender-system" 
LOCAL_DIR = "inference_assets"

# Global variables to hold the loaded assets
finetunedModel = None
xgb_ranker = None
hotel_embeddings = {}
user_factors = np.array([])
item_factors = np.array([])
hotel_df = pd.DataFrame()

# --- 2. Asset Loading on Startup ---
app = FastAPI(title="WanderLust Recommender API")

@app.on_event("startup")
def load_assets():
    global finetunedModel, xgb_ranker, hotel_embeddings, user_factors, item_factors, hotel_df

    print("--- Downloading and loading all assets from GCS on startup ---")
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    # Download all files
    assets_to_download = [
        "processed/xgbScorer.joblib",
        "processed/newEmbedding.npy",
        "processed/userFactor.npy",
        "processed/hotelFactor.npy",
        "processed/combined_hotel_reviews.parquet"
    ]
    for asset_path in assets_to_download:
        blob = bucket.blob(asset_path)
        local_path = os.path.join(LOCAL_DIR, os.path.basename(asset_path))
        blob.download_to_filename(local_path)

    # Download fine-tuned model folder
    sbert_local_path = os.path.join(LOCAL_DIR, "hotel_recommender_finetuned")
    if not os.path.exists(sbert_local_path):
        os.makedirs(sbert_local_path)
    gcs_sbert_path = f"gs://{BUCKET_NAME}/processed/hotel_recommender_finetuned/"
    command = ["gsutil", "-m", "cp", "-r", gcs_sbert_path + "*", sbert_local_path]
    subprocess.run(command, check=True)

    print("Fine-tuned model downloaded successfully.")

    # Load assets into global variables
    xgb_ranker = joblib.load(os.path.join(LOCAL_DIR, 'xgbScorer.joblib'))
    finetunedModel = SentenceTransformer(sbert_local_path)
    hotel_embeddings = np.load(os.path.join(LOCAL_DIR, 'newEmbedding.npy'), allow_pickle=True).item()
    user_factors = np.load(os.path.join(LOCAL_DIR, 'userFactor.npy'))
    item_factors = np.load(os.path.join(LOCAL_DIR, 'hotelFactor.npy'))
    hotel_df = pd.read_parquet(os.path.join(LOCAL_DIR, 'combined_hotel_reviews.parquet'))
    
    print("--- All models and data are loaded and ready. ---")

# --- 3. Recommendation Logic ---
def get_recommendations_final(query, user_id=None, city=None, country=None, top_n=5):
    """
    This function assumes all the assets loaded from GCS bucket and returns a panadas dataframe
    including # top hotels specified by user. The default number is 5. User can be either logged-in
    or in guest mode. Function will use the hybrid model if user is logged in and will use the content-
    based embeddings with fine-tuned model if user is guest. If city is specified by user,
    a prefiltering is applied on the dataset of hotels information dataframe.
    
    Args:
        query(str): The query of user to find top hotels per his taste
        user_id(int, optional): ID of user if he is logged-in which is optional
        city(str, optional): Preferred city for recommendations which is optional
        top_n(int): Number of hotels to be recommended which is defaulted to 5
    
    Returns: 
        pandas.DataFrame: Dataframe of top hotels to be recommended
    """
    # --- Candidate Selection by Location Filtering ---
    # Start with all hotels that have a rating.
    candidate_df = hotel_df[hotel_df['reviews.rating'].notnull()].copy()
    
    if city:
        candidate_df = candidate_df[candidate_df['city'].str.lower() == city.lower()]
    if country:
        candidate_df = candidate_df[candidate_df['country'].str.lower() == country.lower()]
        
    if candidate_df.empty:
        print("No hotels found for the specified location.")
        return pd.DataFrame()

    candidate_hotel_ids = candidate_df['hotel_id'].unique()

    # --- Candidate Generation (Content-Based Ranker) ---
    # Find the top 100 most semantically relevant hotels based on the query.
    print("Finding semantically relevant candidates...")
    query_embedding = finetunedModel.encode(query)
    
    # Filter embeddings to only include our candidates
    candidate_embeddings = {hid: hotel_embeddings.get(hid) for hid in candidate_hotel_ids if hid in hotel_embeddings}
    
    if not candidate_embeddings:
        print("No valid candidates with content embeddings found.")
        return pd.DataFrame()

    hotel_ids, hotel_embs = list(candidate_embeddings.keys()), list(candidate_embeddings.values())
    
    similarities = cosine_similarity(query_embedding.reshape(1, -1), np.array(hotel_embs))[0]
    
    # Create a DataFrame of semantically similar candidates
    similarity_df = pd.DataFrame({'hotel_id': hotel_ids, 'similarity_score': similarities})
    
    # Get the Top 100 most similar candidates
    l1_candidates_df = similarity_df.sort_values(by='similarity_score', ascending=False).head(100)

    # --- Personalized Re-ranking (Hybrid Ranker) ---
    # If the user is anonymous, we return the top results from the content-based ranker.
    if user_id is None or user_id >= len(user_factors):
        print("Anonymous user. Returning top content-based results.")
        top_recommendations = l1_candidates_df.head(top_n)
        
    # If the user is logged in, we re-rank the 100 candidates for personalization.
    else:
        print(f"Logged-in user ({user_id}). Re-ranking candidates for personalization...")
        
        # Get the feature vectors for only the top 100 candidates
        re_rank_candidates = l1_candidates_df['hotel_id'].tolist()
        recommendation_data = []
        user_svd_vector = user_factors[user_id]
        
        for hotel_id in re_rank_candidates:
            hotel_svd_vector = item_factors[hotel_id]
            hotel_content_vector = hotel_embeddings.get(hotel_id)
            feature_vector = np.hstack([user_svd_vector, hotel_svd_vector, hotel_content_vector])
            recommendation_data.append({'hotel_id': hotel_id, 'features': feature_vector})
            
        reco_df = pd.DataFrame(recommendation_data)
        X_reco = np.vstack(reco_df['features'].values)
        
        # Predict with XGBoost
        final_scores = xgb_ranker.predict(X_reco)
        reco_df['score'] = final_scores
        
        # Get the top N from the re-ranked list
        top_recommendations = reco_df.sort_values(by='score', ascending=False).head(top_n)

    # --- Merge and Return Final Results ---
    hotel_info = hotel_df[['hotel_id', 'name', 'city', 'country']].drop_duplicates()
    final_results = pd.merge(top_recommendations, hotel_info, on='hotel_id')
    
    return final_results[['name', 'city', 'country']]

# --- 4. API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Welcome to the WanderLust Recommender API"}

@app.get("/recommend")
def recommend(query: str, user_id: int = None, city: str = None, country: str = None):
    recommendations = get_recommendations_final(query=query, user_id=user_id, city=city, country=country, top_n=5)
    if recommendations.empty:
        return {"message": "No recommendations found for your query."}
    return recommendations.to_dict(orient='records')
