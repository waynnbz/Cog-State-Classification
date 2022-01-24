import pandas as pd
import xgboost as xgb
import sys
import warnings
warnings.filterwarnings('ignore')

from config import NUM_ROUND, PARAM
from utils import load_data_in_chunks, train_preprocess


def main():
    if len(sys.argv) < 2 or len(sys.argv[1]) == 0:
        print("Training input file is missing.")
        return 1

    if len(sys.argv) < 3 or len(sys.argv[2]) == 0:
        print("Training output file is missing.")
        return 1

    print('Training started.')

    input_file = sys.argv[1]
    model_file = sys.argv[2]

    # My Code Starts HERE

    # load & process data
    df = load_data_in_chunks(input_file)
    xg_train = train_preprocess(df)

    # train xgb model
    watchlist = [(xg_train, 'train')]
    num_round = NUM_ROUND
    param = PARAM
    bst = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=10)

    # output trained model
    bst.save_model(model_file)

    print('Training finished.')

    return 0


if __name__ == "__main__":
    main()
