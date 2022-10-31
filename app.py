from ner.components.model_architecture import XLMRobertaForTokenClassification
from ner.config.configurations import Configuration
from ner.exception.exception import CustomException
from ner.pipeline.train_pipeline import TrainPipeline

from typing import Any, Dict, List, ClassVar
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import torch
import os
import sys

app = FastAPI()
class PredictPipeline:
    def __init__(self, config):
        self.predict_pipeline_config = config.get_model_predict_pipeline_config()
        self.tokenizer = self.predict_pipeline_config.tokenizer

        if len(os.listdir(self.predict_pipeline_config.output_dir)) == 0:
            raise LookupError("Model not found : please Run Training Pipeline from pipeline/train_pipeline.py")
        self.model = XLMRobertaForTokenClassification.from_pretrained(pretrained_model_name_or_path = self.predict_pipeline_config.output_dir)

    def run_data_preparation(self, data: str):
        try:
            data = data.split()
            input_ids = self.tokenizer(data, truncation=self.predict_pipeline_config.truncation,
            is_split_into_words=self.predict_pipeline_config.is_split_into_words)
            formatted_data = torch.tensor(input_ids["input_ids"]).reshape(-1, 1)
            outputs = self.model(formatted_data).logits
            predictions = torch.argmax(outputs, dim=-1)
            pred_tags = [self.predict_pipeline_config.index2tag[i.item()] for i in predictions[1:-1]]
            return pred_tags[1:-1]
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self, data):
        predictions = self.run_data_preparation(data)
        response = {
            "Input_Data": data.split(),
            "Tags": predictions
        }
        return response


@app.post("/train")
def train(request: Request):
    try:
        pipeline = TrainPipeline(Configuration())
        pipeline.run_pipeline()
        return JSONResponse(content="Training Completed", status_code=200)
    except Exception as e:
        return JSONResponse(content={"Error": str(e)}, status_code=500)

@app.post("/predict")
def predict(request: Request,data:str):
    try:
        pipeline = PredictPipeline(Configuration())
        response = pipeline.run_pipeline(data)
        return JSONResponse(content=response, status_code=200)
    except Exception as e:
        return JSONResponse(content={"Error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
