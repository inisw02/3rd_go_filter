from typing import Dict
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import Model, get_model

app = FastAPI()

class ClassifierRequest(BaseModel):
    sentence: str

class ClassifierResponse(BaseModel):
    result : str

@app.post("/inference/", response_model=ClassifierResponse)
def inference(request: ClassifierRequest, model: Model = Depends(get_model)):
    result = model.inference(request.sentence)
    return ClassifierResponse(
        result = result
    )
