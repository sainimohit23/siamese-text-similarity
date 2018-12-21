# Siamese Text Similarity
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

A deep RNN siamese network trained to predict similarity score between two texts.

Dataset can be downloaded from [here](http://ixa2.si.ehu.es/stswiki/index.php/Main_Page).

### Dependencies
* Keras
* Numpy
* Time
* os
* pickle

### Model architecture

![untitled diagram](https://user-images.githubusercontent.com/26195811/50359327-5f0bb180-0582-11e9-89ef-73744e359565.jpg)


### Quick Start
* Use `preprocessTrain.py` and `preprocessValid.py` to preprocess data. It uses glove_vectors( [download](https://drive.google.com/open?id=1XSau4ZPwBiiOka-Xwg12Pq0YGMwai8Sz) ) for embedding layer.
* `train.py` trains the model.
* Use `test.py` to test trained model.

### Example output

![screenshot_1](https://user-images.githubusercontent.com/26195811/50359402-a4c87a00-0582-11e9-9967-80743f04cdc5.png)

## References
1. [Learning Text Similarity with Siamese Recurrent Networks](http://www.aclweb.org/anthology/W16-16#page=162)
2. [Siamese Recurrent Architectures for Learning Sentence Similarity (2016)](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12195)
3. [Siamese Recurrent Architectures for Learning Sentence Similarity](http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf)
