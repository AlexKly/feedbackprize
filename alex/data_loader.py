import pandas as pd


TRAIN, TEST, SUBMISSION = 0, 1, 2

class Dataset():
    def __init__(self):
        # Datasets:
        self.dir_dataset = '/srv/dataset/datafbp'
        self.train_path = f'{self.dir_dataset}/train.csv'
        self.test_path = f'{self.dir_dataset}/test.csv'
        self.submission_path = f'{self.dir_dataset}/sample_submission.csv'

    def load_data(self, mode):
        """
        Read data from .csv to DataFrame.
        :param mode: Read train, test or submission DF.
        :return: DataFrame.
        """
        if mode == TRAIN:
            return pd.read_csv(self.train_path)
        elif mode == TEST:
            return pd.read_csv(self.test_path)
        elif mode == SUBMISSION:
            return pd.read_csv(self.submission_path)
        else:
            return None

    @staticmethod
    def prepare_dataset(df):
        df = df.drop(['text_id'], axis=1)

        return df


d = Dataset()
df = d.load_data(mode=TRAIN)
d.prepare_dataset(df=df)
