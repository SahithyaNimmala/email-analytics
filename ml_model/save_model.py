import pandas as pd
import numpy as np
import re

import os 

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from sklearn.model_selection import train_test_split
import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, GlobalMaxPool1D, Conv1D, LSTM, SpatialDropout1D
from keras.layers import Dropout, Bidirectional, BatchNormalization

import emoji
import nltk



data_set = pd.read_csv("training_set_test.csv", names=['target', 'id', 'date', 'flag', 'user', 'text'], encoding = 'latin', header=None)

data_set['target'].replace({4: 1}, inplace=True)


def clean_data(data_set):

   # Remove @ sign
   data_set = re.sub("@[A-Za-z0-9]+", "", data_set)  
   
   # Remove http links
   data_set = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", data_set)  
   data_set = " ".join(data_set.split())
   
   # Remove Emojis
   data_set = ''.join(c for c in data_set if c not in emoji.UNICODE_EMOJI)  
   
   # Remove hashtag sign and keep the text
   data_set = data_set.replace("#", "").replace("_", " ")  
   data_set = " ".join(w for w in nltk.wordpunct_tokenize(data_set))
   return data_set



data_set['text'] = data_set['text'].map(lambda x: clean_data(x))
text = data_set['text']
label = data_set['target']


train_data , test_data , target_train, target_test = train_test_split(text, label,random_state=1000,) 


# max tokens in one sentence
maxlen = 60  
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data)
vocab_size = len(tokenizer.word_index) + 1


# transform sentence  to vector
X_train = tokenizer.texts_to_sequences(train_data)
X_test = tokenizer.texts_to_sequences(test_data)

# padding sequence
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


embedding_dim = 100


def create_embedding_matrix(filepath, word_index, embedding_dim):
   vocab_size = len(word_index) + 1 
   embedding_matrix = np.zeros((vocab_size, embedding_dim))

   with open(filepath) as f:
      for line in f:
         word, *vector = line.split()
         if word in word_index:
            idx = word_index[word] 
            embedding_matrix[idx] = np.array(vector, dtype=np.float32)[:embedding_dim]
   
   return embedding_matrix

embedding_dim_glove = 50
embedding_matrix = create_embedding_matrix('glove.6B.50d.txt', tokenizer.word_index, embedding_dim_glove)


model3 = Sequential()
model3.add(Embedding(vocab_size, embedding_dim_glove, weights=[embedding_matrix], input_length=maxlen, trainable=True))
model3.add(SpatialDropout1D(0.1))
model3.add(Conv1D(128, 5, activation='relu'))
model3.add(Bidirectional(LSTM(4, return_sequences=True)))
model3.add(Conv1D(128, 5, activation='relu'))
model3.add(GlobalMaxPool1D())
model3.add(Dropout(0.25))
model3.add(Dense(10, activation='relu'))
model3.add(Dense(1, activation='sigmoid'))
model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model3.summary()


history3 = model3.fit(X_train, target_train, epochs=15, verbose=1, validation_data=(X_test, target_test), batch_size=150)


def transform_sentence_to_sequence(sentence, tokenizer, maxlen):
   texts = tokenizer.texts_to_sequences([sentence])
   texts = pad_sequences(texts, padding='post', maxlen=maxlen)
   return texts

pred = model3.predict([transform_sentence_to_sequence("i am sad  with you services ", tokenizer, maxlen=maxlen)])


model3.save('lstmModel')
print('saved')