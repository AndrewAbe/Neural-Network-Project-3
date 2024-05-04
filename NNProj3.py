import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

dataset_path = "C:\\Users\\raemu\\Desktop\\Cal State Fullerton\\Spring 2024\\585 Neural Networks\\Project 3\\RateMyProfessor_Sample data.csv"
df = pd.read_csv(dataset_path)

# Preprocess the dataset
df = df.dropna(subset=['comments', 'student_star', 'student_difficult'])

# Extract text and labels
comments = df['comments'].values
quality_labels = df['student_star'].values
difficulty_labels = df['student_difficult'].values

# Tokenization
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(comments)
sequences = tokenizer.texts_to_sequences(comments)
maxlen = 100  # Assuming a maximum sequence length of 100
data = pad_sequences(sequences, maxlen=maxlen)

x_train, x_test, y_train_quality, y_test_quality, y_train_difficulty, y_test_difficulty = train_test_split(
    data, quality_labels, difficulty_labels, test_size=0.2, random_state=42)

embeddings_index = {}
with open("C:\\Users\\raemu\\Desktop\\Cal State Fullerton\\Spring 2024\\585 Neural Networks\\Project 3\\glove.6B.100d.txt", encoding='utf-8') as f:
    next(f)  # Skip the header line
    for line in f:
        values = line.split()
        word = values[0]
        if len(values) > 1:
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

# Create an embedding matrix
word_index = tokenizer.word_index
embedding_dim = 100
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# Define the RNN model architecture for quality prediction
quality_model = Sequential()
quality_model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix],
                            input_length=maxlen, trainable=False))
quality_model.add(Bidirectional(LSTM(64)))
quality_model.add(Dense(1, activation='linear'))

# Compile the quality model
quality_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the quality model
quality_model.fit(x_train, y_train_quality, epochs=10, batch_size=32, validation_data=(x_test, y_test_quality))

# Evaluate the quality model
quality_loss, quality_mae = quality_model.evaluate(x_test, y_test_quality)
print("quality prediction:", quality_mae)

# Define the RNN model architecture for difficulty prediction
difficulty_model = Sequential()
difficulty_model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix],
                               input_length=maxlen, trainable=False))
difficulty_model.add(Bidirectional(LSTM(64)))
difficulty_model.add(Dense(1, activation='linear'))

# Compile the difficulty model
difficulty_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the difficulty model
difficulty_model.fit(x_train, y_train_difficulty, epochs=10, batch_size=32, validation_data=(x_test, y_test_difficulty))

# Evaluate the difficulty model
difficulty_loss, difficulty_mae = difficulty_model.evaluate(x_test, y_test_difficulty)
print("difficulty prediction:", difficulty_mae)


""" SECOND PASS
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a long enough 'pe' for max_len tokens
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Sample tokenizer function - you'll likely need something more robust
def tokenize(comments, vocab):
    tokenized = [[vocab[word] for word in comment.lower().split()] for comment in comments]
    return tokenized

# Example of building a simple vocabulary and tokenizer
def build_vocab(comments):
    word_count = defaultdict(int)
    for comment in comments:
        for word in comment.lower().split():
            word_count[word] += 1
    vocab = {word: i+1 for i, word in enumerate(word_count)}  # +1 for zero padding
    return vocab

# Loading your data
comments = ["This is an example comment.", "Another example comment."]
quality_labels = torch.tensor([3.5, 4.0])
difficulty_labels = torch.tensor([2.0, 3.0])

# Build vocabulary from comments
vocab = build_vocab(comments)

# Tokenize comments
tokenized_comments = tokenize(comments, vocab)

# Convert to PyTorch tensors and pad
comments_tensor = pad_sequence([torch.tensor(comment) for comment in tokenized_comments], batch_first=True, padding_value=0)

# Splitting data
train_comments, test_comments, train_quality_labels, test_quality_labels, train_difficulty_labels, test_difficulty_labels = train_test_split(
    comments_tensor, quality_labels, difficulty_labels, test_size=0.2, random_state=42)


class RateMyProfessorDataset(Dataset):
    def __init__(self, comments, quality_labels, difficulty_labels):
        self.comments = comments
        self.quality_labels = quality_labels
        self.difficulty_labels = difficulty_labels

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        return self.comments[idx], self.quality_labels[idx], self.difficulty_labels[idx]

# Splitting data
train_comments, test_comments, train_quality_labels, test_quality_labels, train_difficulty_labels, test_difficulty_labels = train_test_split(
    comments_tensor, quality_labels, difficulty_labels, test_size=0.2, random_state=42)

# Create dataset instances
train_dataset = RateMyProfessorDataset(train_comments, train_quality_labels, train_difficulty_labels)
test_dataset = RateMyProfessorDataset(test_comments, test_quality_labels, test_difficulty_labels)

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class TransformerModel(nn.Module):
    def __init__(self, ntoken, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(nhid, dropout)
        self.encoder = nn.Embedding(ntoken, nhid)
        transformer_layer = nn.TransformerEncoderLayer(d_model=nhid, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=nlayers)
        self.decoder = nn.Linear(nhid, 2)  # Assuming 2 outputs for your model

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(nhid)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output.mean(dim=1))
        return output

# Initialize the model
ntoken = 10000  # Assuming vocabulary size of 10000
nhid = 512  # Embedding dimension
nhead = 8  # Number of heads in multi-head attention models
nlayers = 3  # Number of transformer layers
dropout = 0.5  # Dropout rate

model = TransformerModel(ntoken, nhead, nhid, nlayers, dropout)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for inputs, quality_labels, difficulty_labels in train_loader:
        inputs, quality_labels, difficulty_labels = inputs.to(device), quality_labels.to(device), difficulty_labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_quality = criterion(outputs[:, 0], quality_labels.float())
        loss_difficulty = criterion(outputs[:, 1], difficulty_labels.float())
        loss = loss_quality + loss_difficulty
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Training the model
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
"""





'''
!wget https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
!unzip -q glove.6B.zip

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

dataset_path = '/content/drive/MyDrive/Colab Notebooks/RateMyProfessor_Sample data.csv'
df = pd.read_csv(dataset_path)

# Preprocess the dataset
df = df.dropna(subset=['comments', 'student_star', 'student_difficult'])

# Extract text and labels
comments = df['comments'].values
quality_labels = df['student_star'].values
difficulty_labels = df['student_difficult'].values

# Tokenization
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(comments)
sequences = tokenizer.texts_to_sequences(comments)
maxlen = 100  # Assuming a maximum sequence length of 100
data = pad_sequences(sequences, maxlen=maxlen)

x_train, x_test, y_train_quality, y_test_quality, y_train_difficulty, y_test_difficulty = train_test_split(
    data, quality_labels, difficulty_labels, test_size=0.2, random_state=42)

embeddings_index = {}
with open('/content/glove.6B.100d.txt', encoding='utf-8') as f:
    next(f)  # Skip the header line
    for line in f:
        values = line.split()
        word = values[0]
        if len(values) > 1:
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

# Create an embedding matrix
word_index = tokenizer.word_index
embedding_dim = 100
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# Define the RNN model architecture for quality prediction
quality_model = Sequential()
quality_model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix],
                            input_length=maxlen, trainable=False))
quality_model.add(Bidirectional(LSTM(64)))
quality_model.add(Dense(1, activation='linear'))

# Compile the quality model
quality_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the quality model
quality_model.fit(x_train, y_train_quality, epochs=10, batch_size=32, validation_data=(x_test, y_test_quality))

# Evaluate the quality model
quality_loss, quality_mae = quality_model.evaluate(x_test, y_test_quality)
print("quality prediction:", quality_mae)

# Define the RNN model architecture for difficulty prediction
difficulty_model = Sequential()
difficulty_model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix],
                               input_length=maxlen, trainable=False))
difficulty_model.add(Bidirectional(LSTM(64)))
difficulty_model.add(Dense(1, activation='linear'))

# Compile the difficulty model
difficulty_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the difficulty model
difficulty_model.fit(x_train, y_train_difficulty, epochs=10, batch_size=32, validation_data=(x_test, y_test_difficulty))

# Evaluate the difficulty model
difficulty_loss, difficulty_mae = difficulty_model.evaluate(x_test, y_test_difficulty)
print("difficulty prediction:", difficulty_mae)
'''