import os
from flask import Flask, request
from pydantic import BaseModel

app = Flask(__name__)

class Game(BaseModel):
    id: str
    name: str

@app.get("/games")
def get_all_games():
    return [
        Game(**{
            "id": "1",
            "name": "Game1",
        }).dict(),
        Game(**{
            "id": "2",
            "name": "Game2",
        }).dict(),
        Game(**{
            "id": "3",
            "name": "Game3",
        }).dict(),
    ]

@app.get("/game_prediction")
def get_game_prediction():
    ids = request.args.get('ids')
    if ids:
        ids = ids.split(",")
        return [Game(**{
            "id": "4",
            "name": ids[0],
        }).dict()]
    else:
        return []

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ["API_PORT"]))
