#!/usr/bin/python3
from transformers import PreTrainedTokenizerFast, AutoTokenizer, AutoModel, DistilBertModel,DistilBertConfig
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
        cool = torch.tensor(self.df['cool'].to_list()[i]).squeeze(0)
        funny = torch.tensor(self.df['funny'].to_list()[i]).squeeze(0)
        useful = torch.tensor(self.df['useful'].to_list()[i]).squeeze(0)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': label, 'cool': cool, 'funny': funny, 'useful': useful}

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


class TransformerRNNRegression(nn.Module):
    def __init__(self, device, transformer_model, rnn_hidden_size=256, rnn_layers=2):
        super(TransformerRNNRegression, self).__init__()
        dropout_rate = 0.1
        self.transformer = AutoModel.from_pretrained(transformer_model)
        transformer_hidden_size = self.transformer.config.hidden_size
        self.rnn = nn.LSTM(input_size=transformer_hidden_size,
                           hidden_size=rnn_hidden_size,
                           num_layers=rnn_layers,
                           batch_first=True).to(device)
        
        # Output layer for regression
        self.out =  nn.Sequential(
            nn.Linear(rnn_hidden_size, rnn_hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(rnn_hidden_size * 2, rnn_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(rnn_hidden_size, 1)
        ).to(device)

    def forward(self, input_ids, attention_mask=None):
        transformer_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get the last hidden states
        hidden_states = transformer_output.last_hidden_state
        
        # Pass the hidden states through the RNN
        rnn_output, _ = self.rnn(hidden_states)
        
        # Use the output from the last time step for regression prediction
        last_time_step_output = rnn_output[:, -1, :]
        
        # Pass through the output layer
        output = self.out(last_time_step_output)
        
        return output

#experiment No.1 combare between RNN and CNN 
class DeeperRatingClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Define multiple sets of convolutional layers for each filter size
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(fs, embedding_dim)),
                nn.ReLU(),
                nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(fs, 1)),
                nn.ReLU()
            ) for fs in filter_sizes
        ]).to(device)
        
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, 100)  # Adding an additional dense layer
        self.fc2 = nn.Linear(100, output_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)  # [batch size, 1, sent len, emb dim]
        
        # Apply convolutional blocks
        conved = [conv_block(embedded).squeeze(3) for conv_block in self.conv_blocks]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        cat = self.dropout(torch.cat(pooled, dim=1))
        
        # Pass through additional fully connected layers
        return self.fc2(self.dropout(F.relu(self.fc1(cat))))