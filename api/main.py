import os
from flask import Flask, request
from pydantic import BaseModel
from azure.storage.blob import BlobServiceClient
from autogluon.tabular import TabularPredictor
from dotenv import load_dotenv
import pandas as pd
from io import BytesIO
import zipfile

from test_recommender import get_recommendations

load_dotenv()

app = Flask(__name__)

blob_storage_connection_string = os.environ["BLOB_STORAGE_CONNECTION_STRING"]
model_path_base = os.getcwd()
model_zip_path = os.path.join(model_path_base, "autogluon_recommender.zip")
model_extract_dir = os.path.join(model_path_base, "autogluon_recommender")
enriched_features_path_base = os.getcwd()
enriched_features_zip_path = os.path.join(enriched_features_path_base, "enriched_features.zip")
enriched_features_path = os.path.join(enriched_features_path_base, "enriched_features.parquet")

games = None

class Game(BaseModel):
    id: int
    name: str

def download_and_unzip_model():
    if os.path.isfile(model_zip_path):
        print("Model zip file already exists, skipping download.")
    else:
        print("Downloading model from blob storage...")
        os.makedirs(os.path.dirname(model_zip_path), exist_ok=True)

        blob_service_client = BlobServiceClient.from_connection_string(blob_storage_connection_string)
        blob_client = blob_service_client.get_blob_client(container="model", blob="autogluon_recommender.zip")

        with open(model_zip_path, "wb") as f:
            f.write(blob_client.download_blob().readall())

    if os.path.exists(model_extract_dir):
        print("Model already exists, skipping extraction.")
        return

    print(model_zip_path)
    with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
        zip_ref.extractall(model_path_base)

def load_games_from_blob():
    print("Loading games from blob storage...")

    blob_service_client = BlobServiceClient.from_connection_string(blob_storage_connection_string)
    blob_client = blob_service_client.get_blob_client(container="data", blob="games.csv")

    download_stream = blob_client.download_blob()
    csv_data = download_stream.readall()

    df = pd.read_csv(BytesIO(csv_data))
    return df

def download_and_unzip_recommendations_enriched_from_blob():
    if os.path.isfile(enriched_features_zip_path):
        print("Enriched features zip file already exists, skipping download.")
    else:
        print("Downloading enriched features from blob storage...")
        os.makedirs(os.path.dirname(enriched_features_zip_path), exist_ok=True)

        blob_service_client = BlobServiceClient.from_connection_string(blob_storage_connection_string)
        blob_client = blob_service_client.get_blob_client(container="data", blob="enriched_features.zip")

        with open(enriched_features_zip_path, "wb") as f:
            f.write(blob_client.download_blob().readall())

    if os.path.isfile(enriched_features_path):
        print("Enriched features already exists, skipping extraction.")
        return

    with zipfile.ZipFile(enriched_features_zip_path, 'r') as zip_ref:
        zip_ref.extractall(enriched_features_path_base)

@app.get("/games")
def get_all_games():
    return [game.model_dump() for game in games]

@app.get("/game_prediction")
def get_game_prediction():
    ids = request.args.get('ids')
    if ids:
        id_list = [int(id.strip()) for id in ids.split(",")]
        recommend_similar_games_df = get_recommendations(model_extract_dir, enriched_features_path, id_list, 5)

        if recommend_similar_games_df is None:
            return []

        recommend_similar_games_df = recommend_similar_games_df.rename(columns={"app_id": "id", "title": "name"})
        return [Game(**row).model_dump() for row in recommend_similar_games_df.to_dict(orient="records")]
    else:
        return []

loading_state = "init"

@app.before_request
def startup():
    global loading_state

    if loading_state == "init":
        global games
        loading_state = "loading"

        print("Downloading files...")

        download_and_unzip_model()
        games_df = load_games_from_blob()

        output_games_df = games_df.copy()
        output_games_df = output_games_df.rename(columns={"app_id": "id", "title": "name"})
        games = [Game(**row) for row in output_games_df.to_dict(orient="records")]

        download_and_unzip_recommendations_enriched_from_blob()

        loading_state = "ready"

        print("Files downloaded.")
    elif loading_state == "loading":
        print("Files downloading...")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ["API_PORT"]))
