#!/usr/bin/python3
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from transformers import pipeline
from transformers import TFAutoModel
from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForSequenceClassification
import json
import yelp
import torch
import nltk
import time
from transformers import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainingArguments
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import StepLR
from torch import nn
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report



# def main():
#     file_name = "yelp_academic_dataset_review.json"
#     data_list = []    
#     with open(file_name) as f:
#         lines = f.readlines()[0:10000]
#         for line in lines:
#             data = json.loads(line, object_hook=yelp.Yelp.custom_json_decoder)
#             data_list.append(data.to_dict())
#     print(data_list)
    
#     with open("processed_first_10000.json", "w") as outfile:
#         json.dump(data_list, outfile, indent=4)

model_name = 'distilbert-base-cased'

def validation(model, test_dataset):
    total_correct = 0
    total_count = 0
    val_loss = 0
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    # model = torch.load('TransformerRNNClassifier_Model_800000')
    model.eval()
    validation_loader = DataLoader(test_dataset, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        progress_bar = tqdm(validation_loader)
        for batch in progress_bar:  
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            labels -= 1
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            val_loss += loss.item()
            # Convert logits to predictions
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(classification_report(true_labels, predictions))
    accuracy = accuracy_score(true_labels, predictions)
    avg_loss = val_loss / len(validation_loader)
    print(f"Validation Loss: {avg_loss}, Accuracy: {accuracy}")

def train(file_name):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    file_name = "processed_first_10000.json"
    df = pd.read_json(file_name)
    # train, test = train_test_split(df, test_size=0.3)
    test_file_name = "processed_last_5k.json"
    test_df = pd.read_json(test_file_name)
    train_dataset = yelp.YelpDataset(df)
    test_dataset = yelp.YelpDataset(test_df)
    
    train_loader = DataLoader(train_dataset, shuffle=True)
    model = yelp.TransformerRNNClassifier(device=device,transformer_model_name=model_name, hidden_dim=128, num_layers=3, num_classes=5) 
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Scheduler is optional but can help with learning rate scheduling
    num_epochs = 3
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0, 
                                                num_training_steps=len(train_loader) * num_epochs)
    model.train()
    for epoch in range(num_epochs):
        # Wrap your data loader with tqdm
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}',leave=True)
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target = batch['labels'].to(device)
            target -= 1
            outputs = model(input_ids, attention_mask)  # Forward pass
            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            progress_bar.set_description(f'Epoch {epoch}')
            progress_bar.set_postfix(loss=loss.item())

    torch.save(model, 'FurtherEnhancedRatingClassifier')

    validation(model, test_dataset)