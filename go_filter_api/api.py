from typing import Dict
from fastapi import Depends, FastAPI
from pydantic import BaseModel

from model import Model, get_model

app = FastAPI()

class ClassifierRequest(BaseModel):
    sentence: str

class ClassifierResponse(BaseModel):
    result : str

@app.get("/inference", response_model=ClassifierResponse)
def inference(request: ClassifierRequest, model: Model = Depends(get_model)):
    result = model.inference(request.text)
    return ClassifierResponse(
        result = result
    )