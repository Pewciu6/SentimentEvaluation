from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import os

from data import clean_text, TweetDataset

df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', names=['target', 'id', 'date', 'flag', 'user', 'text'])
df = df[['target', 'text']].copy()
df['target'] = df['target'].replace({0: 0, 4: 1})
df['cleaned_text'] = df['text'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['target'], test_size=0.2, random_state=42
)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128)

train_dataset = TweetDataset(train_encodings, y_train)
test_dataset = TweetDataset(test_encodings, y_test)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    eval_strategy ='epoch',
    logging_dir='./logs',
    save_strategy="epoch"
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')
print("Training complete! Model saved to 'saved_model' directory")