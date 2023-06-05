import joblib
import __main__

from fastapi import FastAPI, Request, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from typing import List
from collections import Counter

from model import Model
from reviews_class_model import ReviewsClassModel
from reviews_topic_model import ReviewsTopicModel
from steam_controller import SteamController

setattr(__main__, "Model", Model)
setattr(__main__, "ReviewsClassModel", ReviewsClassModel)
setattr(__main__, "ReviewsTopicModel", ReviewsTopicModel)
model = joblib.load("model.joblib")
reviews_class_model_ = joblib.load("model_reviews_class_model.joblib")
reviews_class_model_.load_model()
reviews_topic_model_ = joblib.load("model_reviews_topic_model.joblib")
reviews_topic_model_.load_model()
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
    content = model.predict(predicted_vector, game_id, liked, top, offset)
    return JSONResponse(content=jsonable_encoder(content))


@app.get("/recommendations")
async def set_selected_games(request: Request, games_ids: List[int] = Query(None)):
    """Set selected games"""
    vector = model.set_initial_vector(games_ids)
    content = {"vector": vector}
    return JSONResponse(content=jsonable_encoder(content))


@app.get("/reviews/{game_id}")
async def get_reviews(game_id):
    reviews_limit = 5
    reviews_min = 3
    top_words_limit = 20

    data = steam_controller.get_game_reviews(game_id)
    data = data.rename(columns={"review": "text"})
    data = reviews_class_model_.predict(data, "class")
    data = data.loc[data["class"] != "0.0"]
    data = reviews_topic_model_.predict(data, "topic")
    top_words = {
        "pos_top_words": "1.0",
        "neg_top_words": "-1.0",
    }

    for top in top_words:
        words = Counter(
            sum(data[data["class"] == top_words[top]]["text"].str.split().to_list(), [])
        ).most_common(top_words_limit)
        top_words[top] = [i[0] for i in words]

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
    if "1.0" in results:
        results["pros"] = results.pop("1.0")[:reviews_limit]
    if "-1.0" in results:
        results["cons"] = results.pop("-1.0")[:reviews_limit]

    results.update(top_words)
    return results
