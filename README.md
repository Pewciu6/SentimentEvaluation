# Sentiment Analysis App

## Project Structure
```
sentiment-analysis/
├── train.py # File with fine-tuninig process
├── data.py # Data cleansing
├── app.py # FastAPI backend
├── streamlit_app.py # Streamlit frontend
├── saved_model/ # Pretrained DistilBERT model
│ ├── config.json
│ ├── model.safetensors
│ ├── tokenizer_config.json
│ └── vocab.txt
└── README.md
```

## Usage
```
  Start FastAPI backend (in one terminal):
  uvicorn app:app --reload

  Start Streamlit frontend (in another terminal):
  streamlit run streamlit_app.py
```

Access the applications:
```
  FastAPI docs: http://localhost:8000/docs
  Streamlit UI: http://localhost:8501
```
