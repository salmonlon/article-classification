from csv import reader
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer


# from: https://towardsdatascience.com/machine-learning-nlp-text-
# classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
# start
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
# end


class LemmatizerCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(LemmatizerCountVectorizer, self).build_analyzer()
        return lambda doc: (lemmatizer.lemmatize(w) for w in analyzer(doc))


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

# lemmatizing
experiment5_lemmatizing = Pipeline(
    [
        ('vect', LemmatizerCountVectorizer(stop_words='english')),
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegressionCV())
    ]
)

parameters = {
    'vect__binary': (True, False),
    'clf__class_weight': ('balanced', None)
}


def experiment(pipeline, X, y):
    clf = pipeline.fit(X, y)
    cv = cross_val_score(clf, X, y, cv=10)
    acc.append(cv)
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

        # Uncomment the experiment you wish to perform

        # baseline experiment
        print('Bag-of-word')
        experiment(experiment0_baseline, topic_data, y)

        # first experiment
        print('TF-IDF weighting')
        experiment(experiment1_tf_idf, topic_data, y)

        # second experiment
        print('bigram')
        experiment(experiment2_bigram, topic_data, y)

        # third experiment
        # removing stop words
        print('Remove stop words')
        experiment(experiment3_stop_words, topic_data, y)

        # fourth experiment
        print('Stemming')
        stemmer = SnowballStemmer('english', ignore_stopwords=True)
        experiment(experiment4_stem, topic_data, y)

        # fifth experiment
        print('Lemmatizing')
        lemmatizer = WordNetLemmatizer()
        experiment(experiment5_lemmatizing, topic_data, y)



