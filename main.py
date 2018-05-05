import nltk
import numpy as np
import matplotlib.pyplot as plt
from csv import reader
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from util import plot_confusion_matrix


# global variables
label_names = []

'''Experiments are presented in the format of classification pipeline'''
# counts all the occurrence of all the words in all documents
# default setting for vectorizer:
# analyser: single word (other options: char n-grams)
# ngram_range: (1,1) / (min, max)
# stop words: None
# lowercase: True
# max_features: None (limit the number of features by their occurrence)
# binary: True
# vectorizer = CountVectorizer(binary=True)
# generate document-term matrix from training data
# doc_term = vectorizer.fit_transform(X_train)
experiment0 = Pipeline(
    [
        ('vect', CountVectorizer(binary=True)),
        ('clf', LogisticRegressionCV())
    ]
)

# TF-IDF features
experiment1 = Pipeline(
    [
        ('vect', CountVectorizer(binary=False)),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegressionCV())
    ]
)

# todo: find optimal N
# n-gram features
expriment2 = Pipeline(
    [
        ('vect', CountVectorizer(binary=False, ngram_range=(1,3))),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegressionCV())
    ]
)

parameters = {
    'vect__binary': (True, False),
    'clf__class_weight': ('balanced', None)
}


def experiment0_baseline(X, y, cm=False):
    '''
    Baseline experiment
    :param X: Full training data
    :param y: Full labels
    :param cm: display confusion matrix if True,
    :return:
    '''
    clf = experiment0.fit(X, y)
    if cm:
        y_hat = cross_val_predict(clf, X, y, cv=10)
        cm = confusion_matrix(y, y_hat)
        np.set_printoptions(precision=2)
        plt.figure()
        plot_confusion_matrix(cm, classes=label_names, title='Confusion matrix, without normalization')
        plt.show()
    else:
        # 10-fold CROSS VALIDATION
        cv = cross_val_score(clf, X, y, cv=10)
        print("Accuracy: %0.4f (+/- %0.4f)" % (cv.mean(), cv.std() * 2))


def experiment1_tf_idf(X, y):
    clf = experiment1.fit(X, y)
    cv = cross_val_score(clf, X, y, cv=10)
    print("Accuracy: %0.4f (+/- %0.4f)" % (cv.mean(), cv.std() * 2))


def experiment2_2_gram(X, y):
    clf = expriment2.fit(X, y)
    cv = cross_val_score(clf, X, y)
    print("Accuracy: %0.4f (+/- %0.4f)" % (cv.mean(), cv.std() * 2))


if __name__ == '__main__':
    with open('topic.csv', 'r') as f:
        topic_data = list(reader(f))[1:]

        # ground truth labels
        true_labels = [x[0] for x in topic_data]

        # pre-processing raw data
        # merging title and body
        topic_data = [x[2] + ' ' + x[3] for x in topic_data]
        # convert string labels to integer
        le = LabelEncoder()
        y = le.fit_transform(true_labels)
        label_names = le.classes_

        # first experiment
        # experiment0_baseline(topic_data, y)

        # second expriment
        # experiment1_tf_idf(topic_data, y)

        # third expriment
        experiment2_2_gram(topic_data, y)

        # todo: TF-IDF transform

        # classification

        # create pipeline
        # doc_classification = Pipeline(
        #     [('vect', CountVectorizer(binary=True)),
        #      ('clf', LogisticRegressionCV())]
        # )
        #
        # # training the model
        # doc_classification = doc_classification.fit(X_train, y_train)
        #
        # prediction = doc_classification.predict(X_test)
        # acc = accuracy_score(y_test, prediction)
        # print('Accuracy: {0:.2f}'.format(acc))
        #
        # cm = confusion_matrix(y_test, prediction)
        # np.set_printoptions(precision=2)
        # plt.figure()
        # plot_confusion_matrix(cm, classes=label_names, title='Confusion matrix, without normalization')
        # plt.show()


