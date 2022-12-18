from typing import Dict
from fastapi import Depends, FastAPI
from pydantic import BaseModel
import json

from model import Model, get_model

app = FastAPI()

model = Model()

with open("Sentence.json", encoding='utf-8') as json_file:
    Sentence = json.load(json_file)

@app.post("/inference")
def inference(Sentence : str):
    result = model.inference(Sentence)
    return {'result':result}

if __name__ == '__main__':
    uvicorn.run(app, host='local_host', port=8000)