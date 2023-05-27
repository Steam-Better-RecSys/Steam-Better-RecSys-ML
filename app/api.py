import joblib
import __main__
from fastapi import FastAPI, Request, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from typing import List

from model import Model

setattr(__main__, "Model", Model)
model = joblib.load("model.joblib")
app = FastAPI()


@app.get("/health")
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
    """Get new recommendations based on a current game"""
    request = await request.json()
    predicted_vector = request["vector"]
    content = model.predict(predicted_vector, game_id, liked, top, offset)
    return JSONResponse(content=jsonable_encoder(content))


@app.get("/recommendations")
async def set_selected_games(request: Request, games_ids: List[int] = Query(None)):
    """Set selected games"""
    vector = model.set_initial_vector(games_ids)
    content = {"vector": vector}
    return JSONResponse(content=jsonable_encoder(content))
