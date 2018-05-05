'''
This script is used for inspecting the anomalies presented in the classification.
'''

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

        # split training and test data
        X_train, X_test, y_train, y_test = train_test_split(topic_data, y, test_size=0.2)
        n_train = len(X_train)

        # first experiment
        # experiment0_baseline(topic_data, y)

        # second expriment
        # experiment1_tf_idf(topic_data, y)


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

