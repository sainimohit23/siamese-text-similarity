# -*- coding: utf-8 -*-,
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import keras
from keras.preprocessing.text import Tokenizer
import string
import pickle
import gc


def load_sts_data(path):
    """
    Load STS benchmark data.
    """
    genres, sent1, sent2, labels, scores = [], [], [], [], []
    for line in open(path, encoding='utf-8'):
        genre = line.split('\t')[0].strip()
        filename = line.split('\t')[1].strip()
        year = line.split('\t')[2].strip()
        other = line.split('\t')[3].strip()
        score = line.split('\t')[4].strip()
        s1 = line.split('\t')[5].strip()
        s2 = line.split('\t')[6].strip()
        label = float(score)
        genres.append(genre)
        sent1.append(s1)
        sent2.append(s2)
        labels.append(label)
        scores.append(score)
    labels = (np.asarray(labels)).flatten()

    return genres, sent1, sent2, labels, scores

def encode_labels(labels):
    """
    Encode labels Tai et al., 2015
    """
    labels_to_probs = []
    for label in labels:
        tmp = np.zeros(6, dtype=np.float32)
        if (int(label)+1 > 5):
            tmp[5] = 1
        else:
            tmp[int(label)+1] = label - int(label)
            tmp[int(label)] = int(label) - label + 1
        labels_to_probs.append(tmp)
    
    return np.asarray(labels_to_probs)

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

def read_glove_vectors(path):
    """
    read Glove Vector Embeddings
    """
    
    with open(path, encoding='utf8') as f:
        words = set()
        word_to_vec_map = {}
        
        for line in f:
            line = line.strip().split()
            cur_word = line[0]
            words.add(cur_word)
            word_to_vec_map[cur_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map


def get_short_indices(length, sentances):
    """
    Filter out sequences with length "length"
    """
    
    idx = np.zeros((len(sentances)), dtype=bool)   
    for num, sent in enumerate(sentances):
        if len(sent.strip().split()) <= length:
            idx[num] = 1
    
    return idx

vocab_to_int, int_to_vocab, word_to_vec_map = read_glove_vectors('glove.6B.50d.txt')
genres, sent1, sent2, labels, scores = load_sts_data('sts-train.csv')

sent1 = clean_text(sent1)
sent2 = clean_text(sent2)

sent1 = np.array(sent1)
sent2 = np.array(sent2)
labels = np.array(labels)
genres = np.array(genres)


dx = get_short_indices(15, sent1)
sent1 = sent1[dx]
sent2 = sent2[dx]
labels = labels[dx]
genres = genres[dx]

dx = get_short_indices(15, sent2)
sent1 = sent1[dx]
sent2 = sent2[dx]
labels = labels[dx]
genres = genres[dx]

labels = encode_labels(labels)

""" Special Tokens """
codes = ['<PAD>','<EOS>','<UNK>','<GO>']
for code in codes:
    vocab_to_int[code] = len(vocab_to_int)+1

int_to_vocab[len(int_to_vocab)+1] = '<PAD>'
int_to_vocab[len(int_to_vocab)+1] = '<EOS>'
int_to_vocab[len(int_to_vocab)+1] = '<UNK>'
int_to_vocab[len(int_to_vocab)+1] = '<GO>'

word_to_vec_map['<PAD>'] = np.random.random(50)
word_to_vec_map['<GO>'] = np.random.random(50)
word_to_vec_map['<UNK>'] = np.random.random(50)
word_to_vec_map['<EOS>'] = np.random.random(50)



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


pickle.dump((sent1_padded, sent2_padded, leaks, labels, genres, vocab_to_int, int_to_vocab, word_to_vec_map), open('preprocess.p', 'wb'))