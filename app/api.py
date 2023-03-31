import joblib
import uvicorn
import __main__
import numpy as np
from fastapi import FastAPI, Response, Request, Body
from model import Model

setattr(__main__, "Model", Model)
model = joblib.load("model.joblib")
app = FastAPI()


@app.get('/')
def get_home():
    return {"health_check": "OK"}


@app.get("/recommendations")
def get_recommendations(request: Request, response: Response, game_id: int, liked: int = 1, top: int = 10, offset: int = 0):
    predicted_vector = request.cookies.get("vector")
    predictions = model.predict(predicted_vector, game_id, liked, top, offset)
    response.set_cookie(key="vector", value=predictions["vector"])
    return {"items": predictions["recs"]}


@app.post("/selected_games")
async def set_selected_games(request: Request, response: Response):
    games_ids = await request.json()
    games_ids = games_ids['games_ids']
    vector = model.set_initial_pool(games_ids)
    vector = np.array2string(vector.values, precision=2, separator=',', suppress_small=True)
    predictions = model.predict(vector)
    response.set_cookie(key="vector", value=predictions["vector"])
    return predictions["recs"]


@app.get("/selected_games")
def get_selected_games(request: Request, response: Response):
    return request.cookies.get('vector')


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8080)
