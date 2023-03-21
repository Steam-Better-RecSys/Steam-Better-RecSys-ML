import joblib
import uvicorn
import __main__
from fastapi import FastAPI, Response
from model import Model

setattr(__main__, "Model", Model)
model = joblib.load("model.joblib")
app = FastAPI()


@app.get('/')
def get_home():
    return {"health_check": "OK"}


@app.get('/recommendations')
def get_recommendations(game_id: int):
    recommendations = model.predict(game_id)
    d = {
        "items": recommendations,
    }
    return Response(content=str(d).replace("\'", "\""), media_type="application/json")


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8080)
