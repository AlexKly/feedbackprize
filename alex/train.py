from data_loader import Dataset
from model import FeedbackPrizeModel

TRAIN, TEST, SUBMISSION = 0, 1 ,2

class Train():
    def __init__(self):
        pass

    def train(self):
        # Init utils:
        D = Dataset()
        FBM = FeedbackPrizeModel()

        # Preprocessing data frame:
        df_train = D.load_data(mode=TRAIN)
        df_train = D.prepare_dataset(df=df_train)

        # Cross-validation strategy:

