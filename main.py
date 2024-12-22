# main.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import gdown  # for downloading from Google Drive
import os
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch  # Ensure torch is imported
import time

# ------------------------------
# Google Drive Download Helpers
# ------------------------------

@st.cache_resource
def download_file_if_not_exists(url: str, local_path: str):
    """
    Download a file from Google Drive using gdown if it doesn't already exist locally.

    Args:
        url (str): The direct download link for the file on Google Drive.
        local_path (str): The local path where the file should be saved.
    """
    if not os.path.exists(local_path):
        try:
            gdown.download(url, local_path, quiet=False)
        except Exception as e:
            st.error(f"Failed to download {local_path}: {e}")
            st.stop()
    return local_path

# ------------------------------
# Load Models and Data
# ------------------------------

@st.cache_resource
def load_models():
    """
    Load the TF-IDF vectorizer (local), KNN model (Google Drive), combined features (Google Drive),
    movies DataFrame (local), and Sentence Transformer model.

    Returns:
        tfidf (TfidfVectorizer): Loaded TF-IDF vectorizer from local file.
        knn (NearestNeighbors): Loaded KNN model from Google Drive.
        combined_features (np.ndarray): Combined feature vectors from Google Drive.
        df (pd.DataFrame): Movies DataFrame from local file.
        sentence_model (SentenceTransformer): Sentence Transformer model loaded.
    """
    try:
        # ---------------------------------------------------------------------
        # 1. Local Filenames for Small Files (No Google Drive needed)
        # ---------------------------------------------------------------------
        tfidf_local = "TfidfVectorizer_1000.joblib"  # small file, keep in GitHub
        movies_csv_local = "movies_with_embeddings.csv"  # small file, keep in GitHub

        # ---------------------------------------------------------------------
        # 2. Google Drive Direct Download Links for Large Files
        # ---------------------------------------------------------------------
        knn_url = "https://drive.google.com/uc?export=download&id=1e5gou7WKwe-mUEOdVtWHt9Nc2oxzQHgc"
        combined_url = "https://drive.google.com/uc?export=download&id=1AeBnNUaoAnO7NJjSZq95moOwZbVF7Aza"

        # ---------------------------------------------------------------------
        # 3. Local Filenames (to store downloaded Google Drive files)
        # ---------------------------------------------------------------------
        knn_local = "knn_model_low.joblib"
        combined_local = "combined_features_normalized_low.npy"

        # ---------------------------------------------------------------------
        # 4. Download large files (KNN, Combined) if not already local
        # ---------------------------------------------------------------------
        download_file_if_not_exists(knn_url, knn_local)
        download_file_if_not_exists(combined_url, combined_local)

        # ---------------------------------------------------------------------
        # 5. Load the local small files directly
        # ---------------------------------------------------------------------
        tfidf = joblib.load(tfidf_local)
        df = pd.read_csv(movies_csv_local)

        # ---------------------------------------------------------------------
        # 6. Load the newly downloaded large files
        # ---------------------------------------------------------------------
        knn = joblib.load(knn_local)
        combined_features = np.load(combined_local)

        # ---------------------------------------------------------------------
        # 7. Load Sentence Transformer Model
        # ---------------------------------------------------------------------
        sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

        return tfidf, knn, combined_features, df, sentence_model

    except Exception as e:
        st.error(f"Error loading models or data: {e}")
        return None, None, None, None, None

tfidf, knn, combined_features, df, sentence_model = load_models()

if tfidf is None or knn is None or combined_features is None or df is None or sentence_model is None:
    st.stop()  # Stop the app if any model/data failed to load

# ------------------------------
# Helper Functions
# ------------------------------

def extract_unique_terms(df, category):
    """
    Extract unique terms from a specific category in the DataFrame.
    """
    terms = set()
    for entries in df[category].dropna():
        for entry in entries.split(' '):
            term = entry.strip().lower()
            if term:
                terms.add(term)
    return sorted(list(terms))

def preprocess_selection(selection):
    """
    Convert user input into the preprocessed feature name.
    """
    return selection.lower().replace(' ', '')

def recommend_based_on_movies(selected_movies, df, combined_features, knn_model_low, top_n=10):
    """
    Recommend movies based on selected favorite movies.
    """
    indices = []
    for movie in selected_movies:
        idx_list = df[df['title'].str.lower() == movie.lower()].index.tolist()
        if not idx_list:
            st.warning(f"Movie '{movie}' not found in the dataset.")
        else:
            indices.extend(idx_list)

    if not indices:
        st.error("No valid movies selected.")
        return pd.DataFrame()

    mean_vector = combined_features[indices].mean(axis=0)
    mean_vector_normalized = normalize(mean_vector.reshape(1, -1), norm='l2', axis=1)

    n_neighbors = top_n * 10  # Increase to ensure enough unique titles
    distances, neighbors = knn_model_low.kneighbors(mean_vector_normalized, n_neighbors=n_neighbors)

    recommended_titles = []
    selected_movies_lower = [movie.lower() for movie in selected_movies]
    seen_titles = set(selected_movies_lower)
    for idx in neighbors[0]:
        title = df.iloc[idx]['title']
        title_lower = title.lower()
        if title_lower not in seen_titles:
            recommended_titles.append(title)
            seen_titles.add(title_lower)
        if len(recommended_titles) == top_n:
            break

    final_recommendations = df[df['title'].isin(recommended_titles)].drop_duplicates(subset='title')
    final_recommendations = final_recommendations.head(top_n)
    return final_recommendations

def recommend_based_on_text(user_description, sentence_model, combined_features, df, top_n=10):
    """
    Recommend movies based on user-provided textual description using Sentence Transformers.
    """
    if not user_description.strip():
        st.error("Please enter a description of the movie you want to watch.")
        return pd.DataFrame()

    user_vector = sentence_model.encode(user_description)
    user_vector_normalized = normalize(user_vector.reshape(1, -1), norm='l2', axis=1)

    # combined_features might have more columns if you appended them;
    # adjust if your data extends beyond dimension 384 for SentenceTransformer embeddings.
    similarities = cosine_similarity(user_vector_normalized, combined_features[:, :384])
    similarities = similarities.flatten()

    top_indices = similarities.argsort()[::-1]

    recommended_titles = []
    seen_titles = set()
    for idx in top_indices:
        title = df.iloc[idx]['title']
        if title.lower() not in seen_titles:
            recommended_titles.append(title)
            seen_titles.add(title.lower())
        if len(recommended_titles) == top_n:
            break

    final_recommendations = df[df['title'].isin(recommended_titles)].drop_duplicates(subset='title')
    final_recommendations = final_recommendations.head(top_n)
    return final_recommendations

def display_recommendations(recommended_df):
    """
    Display recommended movies in the Streamlit app.
    """
    for idx, row in recommended_df.reset_index(drop=True).iterrows():
        st.markdown(f"### {idx + 1}. {row['title']}")
        if 'overview' in row and not pd.isna(row['overview']):
            with st.expander("See overview"):
                st.write(row['overview'])
        st.write("---")

# ------------------------------
# Extract Unique Terms for Dropdowns
# ------------------------------

unique_genres = extract_unique_terms(df, 'genres')
actors = extract_unique_terms(df, 'actor')
actresses = extract_unique_terms(df, 'actress')
unique_cast = sorted(list(set(actors) | set(actresses)))
unique_directors = extract_unique_terms(df, 'director')

unique_movie_titles = sorted(df['title'].drop_duplicates().unique())
unique_movie_titles_with_prompt = ["Select a movie"] + list(unique_movie_titles)

# ------------------------------
# Streamlit App Layout
# ------------------------------

st.markdown("<h1 style='text-align: center;'>🎥 Movie Recommendation System 🍿</h1>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.header("About")
    st.write("Welcome to the **Movie Recommendation System**!")
    st.write(
        "Get personalized movie recommendations based on your favorite movies or a description of what you'd like to watch.")
    st.write("Navigate through the tabs to choose a recommendation method.")
    st.write("Enjoy your movie journey! 🎬")

tab1, tab2 = st.tabs(["⭐ Favorite Movies", "💬 Describe a Movie"])

with tab1:
    st.header("Recommend Based on Favorite Movies")
    st.write("Select **three** of your favorite movies to receive similar movie recommendations.")

    col1, col2, col3 = st.columns(3)

    with col1:
        favorite_movie_1 = st.selectbox(
            "First favorite movie:",
            unique_movie_titles_with_prompt,
            index=0
        )

    with col2:
        favorite_movie_2 = st.selectbox(
            "Second favorite movie:",
            unique_movie_titles_with_prompt,
            index=0
        )

    with col3:
        favorite_movie_3 = st.selectbox(
            "Third favorite movie:",
            unique_movie_titles_with_prompt,
            index=0
        )

    if st.button("Get Recommendations", key='fav_movies'):
        if (favorite_movie_1 == "Select a movie" or
                favorite_movie_2 == "Select a movie" or
                favorite_movie_3 == "Select a movie"):
            st.error("Please select all three favorite movies.")
        else:
            if len(set([favorite_movie_1, favorite_movie_2, favorite_movie_3])) < 3:
                st.error("Please select three different movies.")
            else:
                with st.spinner('Generating recommendations...'):
                    time.sleep(1)
                    recommendations = recommend_based_on_movies(
                        [favorite_movie_1, favorite_movie_2, favorite_movie_3],
                        df,
                        combined_features,
                        knn,
                        top_n=20
                    )
                if recommendations is not None and not recommendations.empty:
                    st.success("Here are your recommended movies:")
                    display_recommendations(recommendations.reset_index(drop=True))
                    st.balloons()
                else:
                    st.write("No recommendations found based on your selections.")

with tab2:
    st.header("Recommend Based on Description")
    st.write("Describe the kind of movie you want to watch, and we'll recommend movies based on your description.")

    user_description = st.text_area(
        "Enter your movie preferences:",
        height=150,
        placeholder="e.g., A thrilling adventure in space with unexpected twists."
    )

    if st.button("Get Recommendations", key='description'):
        if not user_description.strip():
            st.error("Please enter a description of the movie you want to watch.")
        else:
            with st.spinner('Generating recommendations...'):
                time.sleep(1)
                recommendations = recommend_based_on_text(
                    user_description,
                    sentence_model,
                    combined_features,
                    df,
                    top_n=20
                )
            if recommendations is not None and not recommendations.empty:
                st.success("Here are your recommended movies:")
                display_recommendations(recommendations.reset_index(drop=True))
                st.balloons()
            else:
                st.write("No recommendations found based on your description.")

st.markdown("---")
st.markdown("<h4 style='text-align: center;'>© 2024 Movie Recommendation System</h4>", unsafe_allow_html=True)
