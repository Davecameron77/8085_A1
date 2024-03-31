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

model_name = 'distilbert-base-cased'
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def validation(model, validation_loader):
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


def validation_for_regression(model, validation_loader):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    df = pd.read_json(file_name)
    model.eval()  # Set the model to evaluation mode
    predictions = []
    actuals = []
    with torch.no_grad():  # No need to track gradients during evaluation
        progress_bar = tqdm(validation_loader)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['cool'].to(device)
            
            # Predict
            outputs = model(input_ids,attention_mask)
            
            predictions.extend(outputs.view(-1).tolist())
            actuals.extend(labels.view(-1).tolist())
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)

def training_TransformerRNNClassifier(train_loader):
    model = yelp.TransformerRNNClassifier(device=device,transformer_model_name=model_name, hidden_dim=128, num_layers=3, num_classes=5) 
    model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-5)
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
            target = batch['labels'].to(device).float()
            target -= 1
            outputs = model(input_ids, attention_mask)  # Forward pass
            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            progress_bar.set_description(f'Epoch {epoch}')
            progress_bar.set_postfix(loss=loss.item())
    torch.save(model, 'TransformerRNNClassifier')
    
def training(train_loader, target):
    model = yelp.TransformerRNNRegression(device, model_name)
    model = model.to(device) 
    loss_fn = torch.nn.MSELoss()
    num_epochs = 3
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    model.train()  # Set model to training mode

    for epoch in range(num_epochs):
        progress_bar = tqdm(train_loader)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target = batch[target].to(device).float()
            target = target
            # Assuming model and data are moved to the same device
            predictions = model(input_ids, attention_mask)
            loss = loss_fn(predictions, target.unsqueeze(1))
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f'Epoch {epoch}')
            progress_bar.set_postfix(loss=loss.item())
    
    torch.save(model, 'TransformerRNNRegressor')

   

