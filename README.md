# Article classification

## Dependency

 - nltk
 - sklearn
 - python 3.6

## Related files

 - topic\_classification.py
    - perform topic classification task
    - execute all the experiments in order
    - report 10-fold cross validation accuracy and std
 - virality\_classification.py
    - perform virality classification task
    - execute all the experiments in order
    - report 10-fold cross validation accuracy and std
 - util.py 
    - containig utility functions
        - plot_confusion_matrix
 - preprocessor.py
    - NLTKPreprocessor used in experiment 5
 - inspect.ipynb
    - Experiment file for document inspecting and calculating related evaluation metrics

## How to run

 - To perform topic classification experiments:

    `python topic_classification.py`

 - To perform virality classification experiments:

    `python virality_classification.py`
