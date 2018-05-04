import nltk
from csv import reader
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.linear_model import LogisticRegressionCV
from collections import defaultdict, Counter
from itertools import groupby


def bow():
    """
    bag of word feature extraction
    :return: an array of the probability for each label
    """

    with open('topic.csv', 'r') as f:
        topic_data_raw = list(reader(f))[1:]

        # invariant data:
        # true labels
        # label names
        true_labels = [x[0] for x in topic_data_raw]
        label_names = set(true_labels)
        n_train = len(true_labels)

        # preprocessing training data
        train = [(x[0], x[2] + ' ' + x[3]) for x in topic_data_raw]

        feat = []
        for klass, text in train:
            token_counts = feature_extraction(text)
            feat.append((klass, token_counts))

        # approach 2: title weighted more than body

        # 10-fold cross validation
        # ShuffleSplit avoid dependency within batch data
        cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
        # accuracy = cross_val_score(bow, data, true_labels, cv=cv)
    return


def feature_extraction(text):
    # stop words
    # stops = set(nltk.corpus.stopwords.words('english'))

    tokens = text.lower().split()
    tokens = [tok for tok in tokens if tok.isalpha() and tok not in stops]
    return Counter(tokens)


if __name__ == '__main__':
    bow()

