import os
from os.path import dirname, join, abspath
import json
import pickle
from collections import Counter
import re

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

import tensorflow as tf

import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input, Embedding, Dense, Lambda, LSTM
from keras.layers.pooling import _GlobalPooling1D, GlobalAveragePooling1D
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import get_file

ABS_PATH = abspath(__file__)
FOLDER_PATH = abspath(join(ABS_PATH,os.pardir,'chatter'))

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(join(PROJECT_PATH,"data/conversations.txt")) as f:
    content = f.readlines()
content = [x.strip() for x in content] 

questions = []
answers = []

for i in range(len(content)-3):
    if len(json.loads(content[i]))%2 == 0 and len(json.loads(content[i])) > 0:
        for j in range(0,len(json.loads(content[i])),2):
            questions.append(json.loads(content[i])[j]['text'])
            answers.append(json.loads(content[i])[j+1]['text'])
            

tk = pickle.load(open(join(PROJECT_PATH,"bin/ru_tokenizer_dssm"), "rb"))

print("Токенайзер загружен")

def dummy_loss(y_true, y_pred):
    return y_pred

class MaskedGlobalMaxPooling1D(_GlobalPooling1D):
    
    def __init__(self, **kwargs):
        self.support_mask = True
        super(MaskedGlobalMaxPooling1D, self).__init__(**kwargs)
        
    def build(self, input_shape):
        super(MaskedGlobalMaxPooling1D, self).build(input_shape)
        self.feat_dim = input_shape[2]

    def call(self, x, mask=None):
        ans = K.max(x, axis=1)
        return ans

    def compute_mask(self, input_shape, input_mask=None):
        return None

model = load_model(join(PROJECT_PATH,"bin/rnn.model"), custom_objects={"MaskedGlobalMaxPooling1D": MaskedGlobalMaxPooling1D, 
                                                  "tf": tf, 
                                                  "dummy_loss": dummy_loss})

print("DSSM загружена")

q_model = Model(inputs=[model.get_layer("QInput").input], 
                outputs=[model.get_layer("QL2norm").output])
a_model = Model(inputs=[model.get_layer("AInput").input], 
                outputs=[model.get_layer("AL2norm").output])
    
    
class QAReplier(object):
    
    def __init__(self, tokenizer, q_model, a_model, **kwargs):
        """
        Constructor
        
        Args:
            tokenizer(keras.Tokenizer): fitted instance of Keras tokenizer class
            q_model(keras.Model):       question encoding tower of DSSM in the form of Keras model
            a_model(keras.Model):       answer encoding tower of DSSM in the form of Keras model
        
        Return:
            self
        """
        
        # call the ancestor constructor
        super(QAReplier, self).__init__(**kwargs)
        # assign passed parameters to the class fields
        self._tk = tokenizer
        self._q_model = q_model
        self._a_model = a_model
        
    def _preprocess(self, bank):
        """
        Preprocess the corpus of sentences into the suitable format
        to feed into the tower.
        
        Args:
            bank(np.ndarray): array of sentences to preprocess
            
        Return:
            bank_padded(np.ndarray): preprocessed and tokenized sentences
        """
        
        bank_processed = []
        # iterate over the corpus
        for i in range(len(bank)):
            # filter out specific symbols, punctuations, 
            # transform numbers into special <num> token, put everything to lower case
            q = " ".join([w if not w.isdigit() and not bool(re.search(r'[a-zA-Z]', w))
                          else "<num>" for w in text_to_word_sequence(bank[i])])
            bank_processed.append(q)
        bank_processed = np.array(bank_processed)
        
        # transform filtered sentences into sequences of tokens
        bank_tokenized = self._tk.texts_to_sequences(bank_processed)
        
        # pad sequences to the desired length (taken from the model input dimensions)
        bank_padded = pad_sequences(bank_tokenized, maxlen=self._q_model.layers[0].input_shape[1])
        
        return bank_padded
        
    def fit(self, answers_bank):
        """
        Fit the replier model to the predefined set of answers.
        All other answers will be chosen from the passed array.
        
        Args:
            answers_bank(np.ndarray): array of sentences from which 
                                      the model can answer

        Return:
            self
        """
        
        self._answers_bank = answers_bank
        # preprocess and then encode answers with the answer tower of DSMM
        _answers_bank_encoded = self._a_model.predict(self._preprocess(self._answers_bank))
        
        # nearest neighbors model to find the most suitable answer
        self._replier = NearestNeighbors(n_neighbors=5, 
                                         metric="minkowski", 
                                         p=2, 
                                         n_jobs=-1, 
                                         algorithm="kd_tree")
        # fir nearest neighbors model on the encoded set of answers
        self._replier.fit(_answers_bank_encoded)
        
        return self
    
    def answer(self, questions, n_answers=5):
        """
        Provide an answer to the given questions.
        
        Args:
            questions(str, list, np.ndarray): questions to the model
            n_answers(int):                   number of top answers to return
            
        Return:
            answers(np.ndarray): array of answers to the given questions
        """
        
        if type(questions) == str:
            questions = np.array([questions])
        elif type(questions) == list:
            questions = np.array(questions)
        elif type(questions) == np.ndarray:
            pass
        else:
            raise ValueError("Wrong format of question")
            
        questions = self._preprocess(questions)
            
        questions_encoded = self._q_model.predict(questions)
        
        res = self._replier.kneighbors(questions_encoded, n_neighbors=n_answers)[1]
        
        return self._answers_bank[res[0][0]],self._answers_bank[res[0][1]],self._answers_bank[res[0][2]], self._answers_bank[res[0][3]], self._answers_bank[res[0][4]]
    


chatter = QAReplier(tk, q_model, a_model)

chatter.fit(answers)

print(chatter.answer("Привет. Как дела?"))

print("Chatter готов к работе")
