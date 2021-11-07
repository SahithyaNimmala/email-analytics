import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import os 
# required  tokenization pickle with word embeddings
train_data = pd.read_pickle(os.path.abspath("main/ml_model/tokenization.pkl"))
model3 = keras.models.load_model(os.path.abspath("main/ml_model/lstm.h5"))
maxlen = 60


def transform_sentence_to_sequence(
    sentence: str, tokenizer: object, maxlen: int
) -> list:
    """[summary]

    Args:
        sentence (str): sentence need to be predicted 
        tokenizer (object): tokenizer the convert to text to text sequence
        maxlen (int): max length of word braking 

    Returns:
        list: [description]
    """
    texts = tokenizer.texts_to_sequences([sentence])
    texts = pad_sequences(texts, padding="post", maxlen=maxlen)
    return texts


def predict_message(text: str) -> dict:
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data)
    pred = model3.predict(
        [transform_sentence_to_sequence(text, tokenizer, maxlen=maxlen)]
    )
    pred =pred.tolist()
    print(pred)

    return {"text": text, "positive": pred[0][0] * 100, "negative": 100 - (pred[0][0] * 100)}
