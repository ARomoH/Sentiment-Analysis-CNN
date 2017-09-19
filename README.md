# Sentiment-Analysis-CNN
Project developed in Python 3.5 making use of Keras library (using TensorFlow as backend) to make a model capable of predicting sentiment polarity associated with Spanish tweets.

## Architecture
The architecture of the Convolutional Neuronal Network developed is the one proposed by [Kim, Y. (2014)](https://arxiv.org/abs/1408.5882)

![Alt text](https://github.com/ARomoH/Sentiment-Analysis-CNN/blob/master/Images/Architecture.png)

## Problem
Model was originally developed to predict Spanish tweets. It was applied to [TASS CORPUS](http://www.sepln.org/workshops/tass/2016/tass2016.php) using word2vect method developed by [Cardellino](http://crscardellino.me/SBWCE/)

## Execution instruction
Inside the code, you must replace Train1_x/Test1_x and Train1_y/Test1_y with the corresponding files. In _x files must be appear the words of all of tweets concatenated using word2vect vectors. While in _y files, it must appear polarity associated to each tweet. The version of libraries used are:
- TensorFlow 1.2.1
- Keras 2.0.6


