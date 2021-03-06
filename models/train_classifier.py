import pickle
import re
import sys
from collections import defaultdict

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sqlalchemy import create_engine

nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'tagsets'])


class MetaData(BaseEstimator, TransformerMixin):

    def count_meta_data(self, text):
        """
        Returns the counts of different meta data based on the text

        Args:
            text (str): The text for which the meta data needs to generated

        Returns:
            dictionary: A dictionary with keys as the different meta data and values as their count
        """

        counter = defaultdict(int)

        # tokenize by sentences
        sentence_list = sent_tokenize(text)

        for sentence in sentence_list:
            # tokenize each sentence into words and tag part of speech
            pos_tags = nltk.pos_tag(word_tokenize(sentence))

            # check each tags word and keep a count of verbs
            for _, tag in pos_tags:
                if tag.startswith('JJ'):
                    counter['adjective'] += 1
                elif tag.startswith('NN'):
                    counter['noun'] += 1
                elif tag.startswith('PRP'):
                    counter['pronoun'] += 1
                elif tag.startswith('RB'):
                    counter['adverb'] += 1
                elif tag.startswith('VB'):
                    counter['verb'] += 1

        return counter

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        """
        Returns a dataframe containing the meta data about the input text

        Args:
            X (numpy.array): A numpy array containing text for which meta data needs to generated

        Returns:
            DataFrame: A dataframe containing the meta data
        """

        # apply count meta data for each text
        X_tagged = pd.Series(X).apply(lambda x: self.count_meta_data(x)).values

        df = pd.DataFrame.from_records(X_tagged)
        df.fillna(0, inplace=True)
        df = df.astype(int)

        return df


def load_data(database_filepath):
    """
    Reads the sql table into a dataframe and separates the feature and the output labels

    Args:
        database_filepath (str): The filepath of the database file

    Returns:
        tuple: Returns a tuple of three elements.
               X: The feature i.e. the messages
               Y: The output labels
               category_names: The names of the several categories
    """

    # create sql alchemy engine
    engine = create_engine(f'sqlite:///{database_filepath}')

    # read the sql table
    df = pd.read_sql_table('DisasterResponse', engine)

    # separating the feature: message
    X = df['message']

    # separating the output labels
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    # extracting the category names
    category_names = list(np.array(Y.columns))

    return X, Y, category_names


def tokenize(text):
    """
    Returns the tokenized form of the input text.

    Args:
        text (str): The input text to be tokenized

    Returns:
        list: The list of tokens extracted from text
    """

    # normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize
    tokens = word_tokenize(text)

    # stemming and stop-words removal
    stemmer = PorterStemmer()
    stop_words = stopwords.words('english')

    tokenized = [stemmer.stem(word) for word in tokens if word not in stop_words]

    return tokenized


def build_model():
    """
    Sets up the pipeline and cross validation for the model

    Returns:
        [GridSearchCV]: The grid search cross validation object
    """

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # couldnt get the custom transformer to work.
    # the pickled file would not work.

    # pipeline = Pipeline([
    #     ('features', FeatureUnion([
    #         ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
    #         ('meta_data', MetaData())
    #     ])),
    #     ('clf', MultiOutputClassifier(RandomForestClassifier()))
    # ])

    # parameters = {
    #     'features__tfidf__min_df': (5, 10),
    #     'clf__estimator__n_estimators': (30, 50)
    # }

    parameters = {
        'tfidf__min_df': (5, 10),
        'clf__estimator__n_estimators': (30, 50)
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_weighted', verbose=3)

    return cv


def evaluate_model(model, x_test, y_test, category_names):
    """
    Returns a dataframe containing the performance metrics of the model

    Args:
        model (transformer): The model to be evaluated
        x_test (DataFrame): The feature dataframe
        y_test (DataFrame): The truth labels for the given features
        category_names (list): The names of the categories
    """

    metrics = list()

    predicted = model.predict(x_test)

    for i, column in enumerate(category_names):
        # for binary columns
        accuracy = accuracy_score(y_test.loc[:, column], predicted[:, i])
        f1 = f1_score(y_test.loc[:, column], predicted[:, i])
        precision = precision_score(y_test.loc[:, column], predicted[:, i])
        recall = recall_score(y_test.loc[:, column], predicted[:, i])

        metrics.append([accuracy, f1, precision, recall])

    df = pd.DataFrame(metrics, index=category_names, columns=['accuracy', 'f1_score', 'precision', 'recall'])

    print(df)
    return


def save_model(model, model_filepath):
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
