import os
from pydantic import BaseModel
import gradio as gr
import requests

api_url = os.environ["API_BASE_URL"]

class Game(BaseModel):
    id: str
    name: str

def fetch[TReturn](api_endpoint: str, params: dict[str, str] = None) -> TReturn:
    response = requests.get(f"{api_url}/{api_endpoint}", headers={ "Content-Type": "application/json" }, params=params)
    if response.status_code == 200:
        json_response = response.json()
        return json_response
    else:
        print(f"Error fetching data from {api_endpoint}: {response.status_code}")
        return None

def fetch_all_games() -> list[Game]:
    json_response = fetch("games")
    if json_response:
        return list(map(
            lambda entry: Game(**entry),
            json_response
        ))
    else:
        return []

def fetch_predictions(ids: list[str]) -> list[Game]:
    json_response = fetch("game_prediction", params={"ids": ",".join(ids)})
    if json_response:
        return list(map(
            lambda entry: Game(**entry),
            json_response
        ))
    else:
        return []



games = fetch_all_games()
choices = list(map(lambda game: (game.name, game.id), games))

def predict(selected_ids: list[str]):
    predictions = fetch_predictions(selected_ids)
    if predictions:
        return f"Predicted games: {', '.join(map(lambda game: game.name, predictions))}"
    else:
        return "No predictions available."

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Dropdown(choices=choices, label="Select games you like", multiselect=True)],
    outputs=["text"],
)

demo.launch(server_name="0.0.0.0", server_port=int(os.environ["FRONTEND_PORT"]))
