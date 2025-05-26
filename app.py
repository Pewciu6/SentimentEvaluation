from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = FastAPI()

class TextInput(BaseModel):
    text: str

def predict_bert(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    return torch.argmax(probs).item()

@app.post("/predict")
def predict(input: TextInput):
    prediction = predict_bert(input.text)
    
    sentiment = "positive" if prediction == 1 else "negative"
    return {"text": input.text, "sentiment": sentiment}

tokenizer = DistilBertTokenizer.from_pretrained('./saved_model')
model = DistilBertForSequenceClassification.from_pretrained('./saved_model')
model.eval()