from typing import Dict
from fastapi import Depends, FastAPI
from pydantic import BaseModel
from typing import Optional, List
import json

from model import Model, get_model

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

origins = ['*'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = Model()


class item_(BaseModel):
    Sentence : str

@app.post("/inference")
def inference(Sentence : item_):
    result = model.inference(Sentence.Sentence)
    result = {'result':result}
    result = json.dumps(result)
    return result
    
    # pass


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)