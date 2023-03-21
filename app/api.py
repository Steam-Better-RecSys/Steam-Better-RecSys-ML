import pickle
import uvicorn
from fastapi import FastAPI, Response
from model import Model

app = FastAPI()


@app.get('/')
def get_home():
    return {"health_check": "OK"}


@app.get('/recommendations')
def get_recommendations(game_id: int):
    recommendations = model.predict(game_id)
    d = {
        "item": str(recommendations[0]),
    }
    return Response(content=str(d).replace("\'", "\""), media_type="application/json")


if __name__ == "__main__":
    model = pickle.load(open('model.pkl', 'rb'))
    uvicorn.run(app, host='127.0.0.1', port=8080)
