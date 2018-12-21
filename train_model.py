import keras
import numpy as np
import pickle
from keras.layers import Bidirectional, LSTM, Input, Dense, BatchNormalization, Dropout
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import os
import time

EMBEDDING_DIM = 50
MAX_SEQUENCE_LENGTH = 15
VALIDATION_SPLIT = 0.1
RATE_DROP_LSTM = 0.17
RATE_DROP_DENSE = 0.25
NUMBER_LSTM = 15
NUMBER_DENSE_UNITS = 50
ACTIVATION_FUNCTION = 'relu'



def load_preprocess(filename):
    with open(filename, mode='rb') as in_file:
        return pickle.load(in_file)

sent1, sent2, leaks, labels, genres, vocab_to_int, int_to_vocab, word_to_vec_map = load_preprocess('preprocess.p')
sent1_valid, sent2_valid, leaks_valid, labels_valid, genres_valid = load_preprocess('preprocessDev.p')

def data_generator(batch_size):
    while True:
        idx = np.random.randint(len(sent1), size= batch_size)
        x1_batch = sent1[idx]
        x2_batch = sent2[idx]
        labels_batch = labels[idx]
        leaks_batch = leaks[idx]
        
        x_data = {
                'seq1_inp': x1_batch,
                'seq2_inp': x2_batch,
                'leaks_inp' : leaks_batch
                }
        y_data = {
                'pred':labels_batch
                }
        
        yield (x_data, y_data)


def pretrained_embedding_layer(word_to_vec_map, words_to_index):
    emb_dim = word_to_vec_map['pen'].shape[0]
    vocab_size = len(words_to_index) + 1
    emb_matrix = np.zeros((vocab_size, emb_dim))
    
    for word, index in words_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]
    
    emb_layer= keras.layers.embeddings.Embedding(vocab_size, emb_dim, trainable= True)
    
    emb_layer.build((None,))
    emb_layer.set_weights([emb_matrix])
    
    return emb_layer



embedding_layer = pretrained_embedding_layer(word_to_vec_map, vocab_to_int)
lstm_layer1 = Bidirectional(LSTM(NUMBER_LSTM, dropout=RATE_DROP_LSTM, recurrent_dropout=RATE_DROP_LSTM, return_sequences=True))
lstm_layer2 = Bidirectional(LSTM(NUMBER_LSTM, dropout=RATE_DROP_LSTM, recurrent_dropout=RATE_DROP_LSTM))
dropout_layer = Dropout(0.5)


seq1_inp = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32', name='seq1_inp')
net1 = embedding_layer(seq1_inp)
net1 = lstm_layer1(net1)
net1 = dropout_layer(net1)
out1 = lstm_layer2(net1)


seq2_inp = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32', name= 'seq2_inp')
net2 = embedding_layer(seq2_inp)
net2 = lstm_layer1(net2)
net2 = dropout_layer(net2)
out2 = lstm_layer2(net2)

leaks_inp = Input(shape=(leaks.shape[1],), name='leaks_inp')
leaks_out = Dense(units=int(NUMBER_DENSE_UNITS/2), activation=ACTIVATION_FUNCTION)(leaks_inp)


merged = concatenate([out1, out2, leaks_out])
merged = BatchNormalization()(merged)
merged = Dropout(RATE_DROP_DENSE)(merged)
merged = Dense(NUMBER_DENSE_UNITS, activation=ACTIVATION_FUNCTION)(merged)
merged = BatchNormalization()(merged)
merged = Dropout(RATE_DROP_DENSE)(merged)

preds = Dense(units=labels.shape[1], activation="softmax", name='pred')(merged)

model = keras.models.Model(inputs=[seq1_inp, seq2_inp, leaks_inp], outputs=preds)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



early_stopping = EarlyStopping(monitor='val_loss', patience=7)
STAMP = 'lstm_%d_%d_%.2f_%.2f' % (NUMBER_LSTM, NUMBER_DENSE_UNITS, RATE_DROP_DENSE, RATE_DROP_LSTM)
checkpoint_dir = './' + 'checkpoints/' + str(int(time.time())) + '/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

bst_model_path = checkpoint_dir + STAMP + '.h5'
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)
tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))

generator = data_generator(64)

model.fit_generator(generator, steps_per_epoch=80, epochs=200, 
                    callbacks=[early_stopping, model_checkpoint, tensorboard],
                    validation_data=([sent1_valid, sent2_valid, leaks_valid],
                                     labels_valid))



with open('bestmodel.txt','w') as file:
    file.write(bst_model_path)




