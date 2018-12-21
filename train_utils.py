import keras
import numpy as np
import pickle
from keras.layers import Bidirectional, LSTM, Input, Dense, BatchNormalization, Dropout
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import os
import time
from keras.preprocessing.sequence import pad_sequences
import re
import string
import gc


def clean_text(lines):
    '''Clean text by removing unnecessary characters and altering the format of words.'''
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    cleaned = list()
    for text in lines:
        text = text.lower()
        
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "that is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"how's", "how is", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"n'", "ng", text)
        text = re.sub(r"'bout", "about", text)
        text = re.sub(r"'til", "until", text)
        text = re.sub(r"[$-()\"#/@;:<>{}`+=~|.!?,'*-]", "", text)
                         
        text = text.split()
        text = [re_print.sub('', w) for w in text]
        
        cleaned.append(' '.join(text))
                         
    return cleaned

def load_preprocess(filename):
    with open(filename, mode='rb') as in_file:
        return pickle.load(in_file)

_, _, _, _, _, vocab_to_int, int_to_vocab, word_to_vec_map = load_preprocess('preprocess.p')


def create_test_data(sent1, sent2):
    sent1 = clean_text(sent1)
    sent2 = clean_text(sent2)
    sent1 = np.array(sent1)
    sent2 = np.array(sent2)
    
    sent1_tokenized = []
    for sent in sent1:
        li= []
        for word in sent.strip().split():
            if word in vocab_to_int.keys():
                li.append(vocab_to_int[word])
            else:
                li.append(vocab_to_int['<UNK>'])
        sent1_tokenized.append(li)
    
    sent2_tokenized = []
    for sent in sent2:
        li= []
        for word in sent.strip().split():
            if word in vocab_to_int.keys(): 
                li.append(vocab_to_int[word])
            else:
                li.append(vocab_to_int['<UNK>'])
        sent2_tokenized.append(li)
        
    del sent1, sent2 #freeing up the memory
    gc.collect()
    
    
    # Keeping track of common words
    leaks = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))] 
            for x1, x2 in zip(sent1_tokenized, sent2_tokenized)]
    leaks = np.array(leaks)
    
    #padding the sequences
    sent1_padded = pad_sequences(sent1_tokenized, maxlen=15,padding='post', truncating='post', value=vocab_to_int['<PAD>'])
    sent2_padded = pad_sequences(sent2_tokenized, maxlen=15,padding='post', truncating='post', value=vocab_to_int['<PAD>'])
    
    del sent1_tokenized, sent2_tokenized #freeing up the memory
    gc.collect()
    
    return (sent1_padded, sent2_padded, leaks)



































