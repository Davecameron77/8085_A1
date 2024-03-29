#!/usr/bin/python3
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

class Yelp:
    review_id = ""
    user_id = ""
    business_id = ""
    stars = 0
    useful = 0
    funny = 0
    cool = 0
    text = ""
    date = ""

    def __init__(self, review_id, user_id, business_id, stars, useful, funny, cool, text, date):
        self.review_id = review_id
        self.user_id = user_id
        self.business_id = business_id
        self.stars = stars
        self.useful = useful
        self.funny = funny
        self.cool = cool
        self.text = text
        self.date = date

    def __str__(self):
        return self.text

    def to_json(self):
        return json.dumps(self, default=lambda o: o.to_dict(), 
            sort_keys=True, indent=4)

    def to_dict(self):
        token = self.tokenize_text()
        print(self.text)
        return {
            # "review_id" : self.review_id,
            # "user_id" : self.user_id,
            # "business_id" : self.business_id,
            "stars" : self.stars,
            "useful" : self.useful,
            "funny" : self.funny,
            "cool" : self.cool,
            "text" : self.text,
            "input_ids" : token[0],
            "attention_mask" : token[1]
            # "date" : self.date,
        }

    def tokenize_text(self):
        text = self.text.lower
        model_name = "distilbert-base-cased"  # Example model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        inputs = tokenizer(self.text, padding="max_length", truncation=True)
        return inputs.input_ids, inputs.attention_mask

    @staticmethod
    def custom_json_decoder(dict):
        review_id = ""
        user_id = ""
        business_id = ""
        stars = 0
        useful = 0
        funny = 0
        cool = 0
        text = ""
        date = ""
        if "review_id" in dict:
            review_id = dict["review_id"]
        if "user_id" in dict:
            user_id = dict["user_id"]
        if "business_id" in dict:
            business_id = dict["business_id"]
        if "stars" in dict:
            stars = dict["stars"]
        if "useful" in dict:
            useful = dict["useful"]
        if "funny" in dict:
            funny = dict["funny"]
        if "cool" in dict:
            cool = dict["cool"]
        if "text" in dict:
            text = dict["text"]
        if "date" in dict:
            date = dict["date"]

        return Yelp(review_id, user_id, business_id, stars, useful, funny, cool, text, date)


class YelpDataset(Dataset):

    def __init__(self, df):
        self.df = df

    def __getitem__(self, i):
        input_ids = torch.tensor(self.df['input_ids'].to_list()[i]).squeeze(0)
        attention_mask = torch.tensor(self.df['attention_mask'].to_list()[i]).squeeze(0)
        label = torch.tensor(self.df['stars'].to_list()[i]).squeeze(0)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label}

    def __len__(self):
        return len(self.df)

    def extract_hidden_states(batch):
        # Place model inputs on the GPU
        inputs = {k:v.to(device) for k,v in batch.items()
                  if k in tokenizer.model_input_names}
        # Extract last hidden states
        with torch.no_grad():
            last_hidden_state = model(**inputs).last_hidden_state
        # Return vector for [CLS] token
        return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}    

class TransformerRNNClassifier(nn.Module):
    def __init__(self, device, transformer_model_name, hidden_dim, num_layers, num_classes):
        super(TransformerRNNClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained(transformer_model_name)
        dropout_rate = 0.1
        self.lstm = nn.LSTM(self.transformer.config.hidden_size, hidden_dim, num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        ).to(device)
    
    
    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            transformer_out = self.transformer(input_ids, attention_mask=attention_mask)
        
        # Using the last hidden state from the transformer
        sequence_output = transformer_out.last_hidden_state
        lstm_out, (hidden, cell) = self.lstm(sequence_output)
        
        pooled_output = lstm_out[:, -1, :]
        # Making predictions at each step
        logits = self.classifier(pooled_output)
        return logits

def tokenize(file_name):
    file_name = file_name
    data_list = []    
    with open(file_name) as f:
        lines = f.readlines()
        lines.reverse()
        lines = lines[0:5000]
        for line in lines:
            data = json.loads(line, object_hook=yelp.Yelp.custom_json_decoder)
            data_list.append(data.to_dict())
    print(data_list)
    
    with open("processed_last_5k.json", "w") as outfile:
        json.dump(data_list, outfile, indent=4)
