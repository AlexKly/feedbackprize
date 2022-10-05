import re
import pandas as pd


class TextProcessor():
    def __init__(self):
        # Preprocessing instruments:
        self.incorrect_substitutions = ['.The', '.I', '.For']
        self.correct_substitutions = ['. The', '. I', '. For']

    @staticmethod
    def load_data(filename):
        """Load data from CSV to Pandas DataFrame.
        :param filename: path to file CSV
        :return: DataFrame from CSV.
        """
        return pd.read_csv(filename)

    def preprocess(self, text):
        """
        :param text: input text.
        :return: preprocessed text.
        """
        # Put space between word and point:
        for i in range(len(self.incorrect_substitutions)):
            text = text.replace(self.incorrect_substitutions[i], self.correct_substitutions[i])
        # Remove extra spaces:
        text = re.sub(r' +', ' ', text)

        return text

    def do_preprocessing(self, filename):
        """
        :param filename:
        :return:
        """
        df = TextProcessor.load_data(filename=filename)
        text = df.apply(self.preprocess)

        return text
