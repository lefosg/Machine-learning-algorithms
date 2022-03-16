#modules import
print("Importing modules")
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm  #lib for cli progress bar
from time import time
from gc import collect
from learning_curves import *  #learning curves and graph functions
#algorithm imports
from NaiveBayes import *
from ID3 import *
from Adaboost import *

#CONSTANTS
_NUM_WORDS = 1000

#FUNCTIONS
def getDataPerPercentage(X, Y, p): 
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=1-p)
    return (xtrain, ytrain), (xtest, ytest)


#Data load
print("Loading data")
(xtrain, ytrain), (xtest, ytest) = tf.keras.datasets.imdb.load_data(num_words=_NUM_WORDS)

word2index = tf.keras.datasets.imdb.get_word_index()

index2word = dict((i + 3, word) for (word, i) in word2index.items())
index2word[0] = '[pad]'  #padding
index2word[1] = '[bos]'  #begining of sentence
index2word[2] = '[oov]'  #out of vocabulary

xtrain = np.array([' '.join([index2word[idx] for idx in text]) for text in xtrain])
xtest = np.array([' '.join([index2word[idx] for idx in text]) for text in xtest])

#CREATE THE VOCABULARY

print("Creating the vocabulary")
vocabulary = list()
for text in xtrain:
    tokens = text.split()
    vocabulary.extend(tokens)

vocabulary = list(set(vocabulary))

#preprocess the vocabulary
print("Preprocessing the vocabulary")
#remove whitespaces from words
for i in range(len(vocabulary)):
    vocabulary[i] = vocabulary[i].strip()

vocabulary = list(set(vocabulary))

#remove numbers that are strings and words with length <= 1
for word in vocabulary:
    if word.isnumeric() or len(word) <= 1:
        vocabulary.remove(word)

#Neutral words exist in the dataset, eg. 'i', 'it' etc. We use the nltk library which already includes some neutral words, to remove them 
from nltk.corpus import stopwords
for stopword in stopwords.words('english'):
    if stopword in vocabulary:
        for _ in range(vocabulary.count(stopword)):
            vocabulary.remove(stopword)


#CREATE BINARY VECTORS

print("Binarizing vectors")
xtrain_binary = list()
xtest_binary = list()

for text in tqdm(xtrain):
    tokens = text.split()
    binary_vector = list()
    for vocab_token in vocabulary:
        if vocab_token in tokens:
            binary_vector.append(1)
        else:
            binary_vector.append(0)
    xtrain_binary.append(binary_vector)

xtrain_binary = np.array(xtrain_binary)

for text in tqdm(xtest):
    tokens = text.split()
    binary_vector = list()
    for vocab_token in vocabulary:
        if vocab_token in tokens:
            binary_vector.append(1)
        else:
            binary_vector.append(0)
    xtest_binary.append(binary_vector)

xtest_binary = np.array(xtest_binary)

#Concatenating data in order to split it later
X = np.concatenate((xtrain_binary, xtest_binary))
Y = np.concatenate((ytrain, ytest))


percentages =  [pr/10 for pr in range(1, 10)]
train_algo_accuracy = np.empty(len(percentages))
train_algo_precision = np.empty(len(percentages))
train_algo_recall = np.empty(len(percentages))
train_algo_f_measure = np.empty(len(percentages))

test_algo_accuracy= np.empty(len(percentages))
test_algo_precision = np.empty(len(percentages))
test_algo_recall =np.empty(len(percentages))
test_algo_f_measure = np.empty(len(percentages))

#Initiate algorithm
print()
a = int(input("Choose an algorithm to run:\n1) Naive Bayes\n2) ID3\n3) Adaboost\nInput: "))
if a < 1 or a > 3:
    print("Invalid algorithm id given. Program terminated")
    exit(-1)
if a == 1:
    algo = NaiveBayesClassification(xtrain_binary, ytrain)
elif a == 2:
    algo = ID3()
#nb = NaiveBayesClassification(xtrain_binary, ytrain)
i = 0
print()
for train_size in percentages:
    (xtrain_binary, ytrain), (xtest_binary, ytest) = getDataPerPercentage(X,Y,train_size)

    if a == 3:
        ab = Adaboost(5, xtrain_binary, ytrain)  #Initialize adaboost

    print("Training with " + str(train_size*100) + "% of the data")
    start = time()
    if a == 1:
        algo.fit()  #Naive Bayes training
    elif a == 2:
        result = algo.fit(xtrain_binary, ytrain, vocabulary)  #ID3 training
    else:
        ab.fit()  #Adaboost training
    print("Testing with " + str(train_size*100) + "% of the data")
    if a == 1:
        train_predictions = algo.predict(xtrain_binary)  #Naive Bayes testing
        test_predictions = algo.predict(xtest_binary)  #Naive Bayes testing
    elif a == 2:
        train_predictions = algo.predict(xtrain_binary, result, vocabulary)  #ID3 testing
        test_predictions = algo.predict(xtest_binary, result, vocabulary)  #ID3 testing
    else:
        train_predictions = ab.predict(xtrain_binary)  #Adaboost testing
        test_predictions = ab.predict(xtest_binary)  #Adaboost testing
    end = time()
    print(f"Total time: {end-start:.2f} sec")

    train_algo_accuracy[i] = accuracy(train_predictions, ytrain)
    train_algo_precision[i] = precision(train_predictions, ytrain)
    train_algo_recall[i] = recall(train_predictions, ytrain)
    train_algo_f_measure[i] = F_measure(train_predictions, ytrain)

    test_algo_accuracy[i] = accuracy(test_predictions, ytest)
    test_algo_precision[i] = precision(test_predictions, ytest)
    test_algo_recall[i] = recall(test_predictions, ytest)
    test_algo_f_measure[i] = F_measure(test_predictions, ytest)

    i += 1
    del xtrain_binary, ytrain, xtest_binary, ytest
    collect()
    print()

algorithm_names = ["Naive Bayes", "ID3", "Adaboost"]
curves = ["Accuracy", "Precision", "Recall", "F1"]
generateGraph(percentages, train_algo_accuracy, curves[0], algorithm_names[a-1], "training", _NUM_WORDS)
generateGraph(percentages, train_algo_precision, curves[1], algorithm_names[a-1], "training", _NUM_WORDS)
generateGraph(percentages, train_algo_recall, curves[2], algorithm_names[a-1], "training", _NUM_WORDS)
generateGraph(percentages, train_algo_f_measure, curves[3], algorithm_names[a-1], "training", _NUM_WORDS)

generateGraph(percentages, test_algo_accuracy, curves[0], algorithm_names[a-1], "testing", _NUM_WORDS)
generateGraph(percentages, test_algo_precision, curves[1], algorithm_names[a-1],"testing", _NUM_WORDS)
generateGraph(percentages, test_algo_recall, curves[2], algorithm_names[a-1], "testing", _NUM_WORDS)
generateGraph(percentages, test_algo_f_measure, curves[3], algorithm_names[a-1], "testing", _NUM_WORDS)

print("Program completed")
input("Press enter to exit...")
