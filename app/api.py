import joblib
import uvicorn
import requests
import __main__
import numpy as np
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from model import Model

setattr(__main__, "Model", Model)
model = joblib.load("model.joblib")
app = FastAPI()


@app.get("/")
def get_home():
    """Check API status"""
    status = {"health_check": "OK"}
    return JSONResponse(content=jsonable_encoder(status))


@app.post("/recommendations")
async def get_recommendations(
    request: Request,
    game_id: int = 0,
    liked: int = 1,
    top: int = 10,
    offset: int = 0,
):
    """Get new recommendations based on a current games"""
    request = await request.json()
    predicted_vector = request["vector"]
    content = model.predict(predicted_vector, game_id, liked, top, offset)
    return JSONResponse(content=jsonable_encoder(content))


@app.post("/selected_games")
async def set_selected_games(request: Request):
    """Set selected games"""
    request = await request.json()
    games_ids = request["games_ids"].split(",")
    vector = model.set_initial_vector(games_ids)
    vector = np.array2string(
        vector.values, precision=2, separator=",", suppress_small=True
    )
    content = {"vector": vector}
    return JSONResponse(content=jsonable_encoder(content))


@app.get('/test')
async def test():
    res = requests.get('https://store.steampowered.com/api/appdetails?appids=730&l=english')
    description = res.json()['730']['data']['detailed_description']
    content = {"description": description}
    return JSONResponse(content=jsonable_encoder(content))



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
