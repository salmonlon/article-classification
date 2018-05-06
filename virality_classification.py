#!/usr/bin/env python -W ignore::DeprecationWarning

from csv import reader
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from nltk.stem.snowball import SnowballStemmer
from preprocessor import NLTKPreprocessor


# from: https://towardsdatascience.com/machine-learning-nlp-text-
# classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
# start
class StemmedCountVectorizer(CountVectorizer):
    """
    CountVectorizer wrapper for performing stemming
    """
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
# end


def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg


def experiment(pipeline, X, y):
    """
    perform 10-fold cross validation experiment using pipeline setup
    :param pipeline: experiment setup
    :param X: raw document data
    :param y: corresponding labels
    """
    clf = pipeline.fit(X, y)
    cv = cross_val_score(clf, X, y, cv=10)
    acc.append(cv)
    print("Accuracy: %0.4f (+/- %0.4f)" % (cv.mean(), cv.std() * 2))


# global variables
label_names = []
acc = []

'''Experiments are presented in the format of classification pipeline'''
# counts all the occurrence of all the words in all documents

# default setting for vectorizer:
# analyser: single word (other options: char n-grams)
# ngram_range: (1,1) / (min, max)
# stop words: None
# lowercase: True
# max_features: None (limit the number of features by their occurrence)
# binary: False
# min_df: 1, discard low frequency words
experiment0_baseline = Pipeline(
    [
        ('vect', CountVectorizer()),
        ('clf', LogisticRegressionCV())
    ]
)

# TF-IDF features
experiment1_tf_idf = Pipeline(
    [
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegressionCV())
    ]
)

# todo: find optimal N
# n-gram features
experiment2_bigram = Pipeline(
    [
        ('vect', CountVectorizer(ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegressionCV())
    ]
)

# remove stop words
experiment3_stop_words = Pipeline(
    [
        ('vect', CountVectorizer(stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegressionCV())
    ]
)

# stemming
experiment4_stem = Pipeline(
    [
        ('vect', StemmedCountVectorizer(stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegressionCV())
    ]
)

# POS lemmatization
experiment5_pos = Pipeline(
    [
        ('vect', NLTKPreprocessor()),
        ('tfidf', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)),
        ('clf', LogisticRegressionCV())
    ]
)


if __name__ == '__main__':
    with open('virality.csv', 'r') as f:
        virality_data = list(reader(f))[1:]

        # ground truth labels
        true_labels = [x[0] for x in virality_data]

        # pre-processing raw data
        # merging title and body
        virality_data = [x[2] + ' ' + x[3] for x in virality_data]
        # convert string labels to integer
        le = LabelEncoder()
        y = le.fit_transform(true_labels)
        label_names = le.classes_

        # baseline experiment
        print('Bag-of-word')
        experiment(experiment0_baseline, virality_data, y)
        #
        # # first experiment
        print('TF-IDF weighting')
        experiment(experiment1_tf_idf, virality_data, y)
        #
        # # second experiment
        print('bigram')
        experiment(experiment2_bigram, virality_data, y)
        #
        # # third experiment
        # # removing stop words
        print('Remove stop words')
        experiment(experiment3_stop_words, virality_data, y)
        #
        # # fourth experiment
        print('Stemming')
        stemmer = SnowballStemmer('english', ignore_stopwords=True)
        experiment(experiment4_stem, virality_data, y)

        # fifth experiment
        experiment(experiment5_pos, virality_data, y)



