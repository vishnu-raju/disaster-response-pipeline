from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'tagsets'])


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

    def fit(self, x):
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
