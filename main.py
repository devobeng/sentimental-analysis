from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel

# Model & Tokenizer Load
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# FastAPI App
app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/analyze")
def analyze_sentiment(data: TextInput):
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item() + 1  # 1–5 stars
        confidence = probs[0][pred_class - 1].item()

    return {
        "text": data.text,
        "stars": pred_class,
        "confidence": round(confidence, 4)
    }
