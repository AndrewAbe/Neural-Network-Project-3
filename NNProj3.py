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
