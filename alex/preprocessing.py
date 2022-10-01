import re
import pandas as pd


TRAIN, TEST, SUBMISSION = 0, 1, 2

class Moea():
    def __init__(self):
        # Paths:
        self.dir_data = '/srv/dataset/datafbp'
        self.train_path = f'{self.dir_data}/train.csv'
        self.test_path = f'{self.dir_data}/test.csv'
        self.submission_path = f'{self.dir_data}/sample_submission.csv'
        self.incorrect_substitutions = ['.The', '.I', '.For']
        self.correct_substitutions = ['. The', '. I', '. For']

    def load_data(self, type_data):
        """Load data from CSV to Pandas DataFrame.
        :param type_data: type data to loading (train, test or submission).
        :return: DataFrame from CSV.
        """
        if type_data == TRAIN:
            filename = self.train_path
        elif type_data == TEST:
            filename = self.test_path
        elif type_data == SUBMISSION:
            filename = self.submission_path
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

    def get_model(self):
        pass


if __name__ == ('__main__'):
    m = Moea()
    data = m.load_data(type_data=TRAIN)
    preprocessed_data = data.full_text.apply(m.preprocess)
    print(preprocessed_data)