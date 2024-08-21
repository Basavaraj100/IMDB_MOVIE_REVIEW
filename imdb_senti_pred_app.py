import pandas as pd 
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import os

import pandas as pd
import numpy as np

import re

import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from contractions import fix
# nltk.download('punkt')
nltk.download('stopwords')
import streamlit as st






#-------------- Support functions ----------------
#  supportive functions
model_file='IMDB_LSTM.h5'
tokenizer_file='tokenizer.pkl'

#. clean the text

def preprocess_text(text):
    # Expand contractions[example can't--->> cannot]
    text = fix(text)

    #  remove some functuals
    f=['<br /><br />']
    for i in f:
      text=re.sub(i,'',text)

    # Lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))


    # Tokenization
    words = text.split()

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # # Stemming (or you could use Lemmatization here)
    # stemmer = PorterStemmer()
    # words = [stemmer.stem(word) for word in words]

    # Rejoin words into a single string
    preprocessed_text = ' '.join(words)

    return preprocessed_text



# to load the pickle file
import pickle
def load_the_pickle_file(path,file):
    model_path = os.path.join(path, file)
    model = pickle.load(open(file, 'rb'))
    return model

#  to load keras model
def load_keras_model(path,file):
  model_path=os.path.join(path,file)
  k_model=tf.keras.models.load_model(file)
  return k_model



#--------------- Prediction function ----------------
def predict_movie_sentiment(text):
  # cleaning the text
  cleaned_text=preprocess_text(text)
  # load model
  model=load_keras_model(os.getcwd(),model_file)
  pad_lenght=model.input_shape[-1]

  #  load tokenizer
  tokenizer=load_the_pickle_file(os.getcwd(),tokenizer_file)
  #  sequencing and padding
  seq_text=tokenizer.texts_to_sequences([cleaned_text])
  # padding
  if len(seq_text)>pad_lenght:
    final_seq_text=seq_text[-pad_lenght:]
  else:
    final_seq_text=pad_sequences(seq_text,maxlen=pad_lenght,padding='pre')
  
  #  predict from the model
  pred=model.predict(final_seq_text)
  # getting result
  res='Positive' if pred[0][0]>0.5 else 'Negavtive'
  return res,pred



#  ------------- Streamlit user interface -----------

st.title('Wecome to Movie Review Sentiment Prediction App')

text=st.text_input('Please enter the movie review here')

if text:
   if st.button('Predict'):
      res,pred=predict_movie_sentiment(text)

      st.text(f'Reviews is {res} with probability {pred}')
else:
   st.warning('Enter the movie review first')

# text='Movie is nice , good dance'
# res,pred=predict_movie_sentiment(text)
# print(res,pred)
      


