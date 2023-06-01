import joblib
import __main__

from fastapi import FastAPI, Request, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from typing import List

from ml.reviews_class_model import ReviewsClassModel
from steam_controller import SteamController
from ml.recommendations_model import RecommendationModel
from ml.reviews_topic_model import ReviewsTopicModel

setattr(__main__, "RecommendationModel", RecommendationModel)
setattr(__main__, "ReviewsClassModel", ReviewsClassModel)
setattr(__main__, "ReviewsTopicModel", ReviewsTopicModel)
recommendations_model = joblib.load("ml/joblib_classes/recommendations_model.joblib")
reviews_class_model = joblib.load("ml/joblib_classes/reviews_class_model.joblib")
reviews_topic_model = joblib.load("ml/joblib_classes/reviews_topic_model.joblib")
steam_controller = SteamController()
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
    content = recommendations_model.predict(
        predicted_vector, game_id, liked, top, offset
    )
    return JSONResponse(content=jsonable_encoder(content))


@app.get("/recommendations")
async def set_selected_games(request: Request, games_ids: List[int] = Query(None)):
    """Set selected games"""
    vector = recommendations_model.set_initial_vector(games_ids)
    content = {"vector": vector}
    return JSONResponse(content=jsonable_encoder(content))


@app.get("/reviews/{game_id}")
async def get_reviews(game_id):
    reviews_limit = 5
    reviews_min = 5

    data = steam_controller.get_game_reviews(game_id)
    data = data.rename(columns={"review": "text"})
    data = reviews_class_model.predict(data, "class")
    data = data.loc[data["class"] != "0.0"]
    data = reviews_topic_model.predict(data, "topic")
    results = (
        data.groupby(["class", "topic"])["embedding"]
        .count()
        .reset_index()
        .rename(columns={"embedding": "count"})
    )
    results = (
        results[results["count"] > reviews_min]
        .sort_values(["count"], ascending=False)
        .groupby(["class"])["topic"]
        .apply(list)
        .to_dict()
    )
    results["pros"], results["cons"], results["id"] = (
        results.pop("1.0")[:reviews_limit],
        results.pop("-1.0")[:reviews_limit],
        game_id,
    )
    return results
