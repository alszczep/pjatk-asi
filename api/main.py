import os
from flask import Flask, request
from pydantic import BaseModel
from azure.storage.blob import BlobServiceClient
import py7zr
from autogluon.tabular import TabularPredictor
from dotenv import load_dotenv
import pandas as pd
from io import BytesIO

load_dotenv()

app = Flask(__name__)

blob_storage_connection_string = os.environ["BLOB_STORAGE_CONNECTION_STRING"]
extract_dir = os.path.join(os.getcwd(), "model")
recommendations_enriched_path = os.path.join(os.getcwd(), "recommendations_enriched.parquet")

class Game(BaseModel):
    id: int
    name: str

def download_and_unzip_model():
    local_zip_path = os.path.join(os.getcwd(), "automl_model.7z")

    if os.path.exists(extract_dir):
        print("Model already exists, skipping download.")
        return

    os.makedirs(os.path.dirname(local_zip_path), exist_ok=True)

    blob_service_client = BlobServiceClient.from_connection_string(blob_storage_connection_string)
    blob_client = blob_service_client.get_blob_client(container="model", blob="automl_model.7z")

    with open(local_zip_path, "wb") as f:
        f.write(blob_client.download_blob().readall())

    with py7zr.SevenZipFile(local_zip_path, mode='r') as archive:
        archive.extractall(path=extract_dir)

def load_games_from_blob():
    blob_service_client = BlobServiceClient.from_connection_string(blob_storage_connection_string)
    blob_client = blob_service_client.get_blob_client(container="data", blob="games.csv")

    download_stream = blob_client.download_blob()
    csv_data = download_stream.readall()

    df = pd.read_csv(BytesIO(csv_data))
    return df

def download_recommendations_enriched_from_blob():
    if os.path.isfile(recommendations_enriched_path):
        print("Recommendations enriched already exists, skipping download.")
        return

    blob_service_client = BlobServiceClient.from_connection_string(blob_storage_connection_string)
    blob_client = blob_service_client.get_blob_client(container="data", blob="recommendations_enriched.parquet")

    with open(recommendations_enriched_path, "wb") as f:
        f.write(blob_client.download_blob().readall())

def load_recommendations_enriched():
    download_recommendations_enriched_from_blob()
    return pd.read_parquet(recommendations_enriched_path)

download_and_unzip_model()
predictor = TabularPredictor.load(os.path.join(extract_dir, "automl_model"))
games_df = load_games_from_blob()

output_games_df = games_df.copy()
output_games_df = output_games_df.rename(columns={"app_id": "id", "title": "name"})
games = [Game(**row) for row in output_games_df.to_dict(orient="records")]

recommendations_enriched_df = load_recommendations_enriched()

def recommend_similar_games(game_ids: list[int], top_n: int = 10) -> pd.DataFrame:
    all_candidates = (
        recommendations_enriched_df[~recommendations_enriched_df["app_id"].isin(game_ids)]
        .drop_duplicates(subset="app_id")  # ensures each game appears only once
        .copy()
    )

    drop_cols = ["user_id", "app_id", "title", "is_recommended", "date", "review_id"]
    all_candidates = all_candidates.drop(columns=[col for col in drop_cols if col in all_candidates.columns])

    all_candidates = all_candidates.fillna(-1)

    proba_df = predictor.predict_proba(all_candidates)
    scores = proba_df[True]

    top_indices = scores.sort_values(ascending=False).head(top_n).index
    top_games = recommendations_enriched_df.loc[top_indices][["app_id", "title"]].copy()
    top_games["score"] = scores.loc[top_indices].values

    return top_games.reset_index(drop=True)

@app.get("/games")
def get_all_games():
    return [game.model_dump() for game in games]

@app.get("/game_prediction")
def get_game_prediction():
    ids = request.args.get('ids')
    if ids:
        id_list = [int(id.strip()) for id in ids.split(",")]
        recommend_similar_games_df = recommend_similar_games(id_list, top_n=10)
        recommend_similar_games_df = recommend_similar_games_df.rename(columns={"app_id": "id", "title": "name"})
        return [Game(**row).model_dump() for row in recommend_similar_games_df.to_dict(orient="records")]
    else:
        return []

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ["API_PORT"]))
