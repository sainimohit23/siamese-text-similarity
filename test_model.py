import keras
import numpy as np
import pickle
from keras.layers import Bidirectional, LSTM, Input, Dense, BatchNormalization, Dropout
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import os
import time
from train_utils import *

with open('bestmodel.txt', 'r') as file:
    best_model_path = file.read()

model = keras.models.load_model(best_model_path)

def calculate_total_score(arr):
    score = 0
    for i in range(len(arr)):
        score = score + (i*arr[i])

    return score

sent1 = ['What can make Physics easy to learn?','How many times a day do a clocks hands overlap?']
sent2 = ['How can you make physics easy to learn?', 'What does it mean that every time I look at the clock the numbers are the same?']
test_data_x1, test_data_x2, leaks_test = create_test_data(sent1, sent2)

preds = model.predict([test_data_x1, test_data_x2, leaks_test], verbose=1)

print("\n")
for i in range(len(preds)):
    print("Sentance 1: " + sent1[i])
    print("Sentance 2: " + sent2[i])
    print("Similarity Score: " + str(calculate_total_score(preds[i])))
    print("\n")

