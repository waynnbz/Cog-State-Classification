import pandas as pd
import xgboost as xgb
import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

from config import OUTPUT_HEADER
from utils import load_data_in_chunks, test_postprocess

def main():
    
    if len(sys.argv) < 2 or len(sys.argv[1]) == 0:
        print("Testing input file is missing.")
        return 1
    
    if len(sys.argv) < 3 or len(sys.argv[2]) == 0:
        print("Testing output file is missing.")
        return 1
    
    print('Testing started.')

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    model_file = sys.argv[3]
    
    
    # My Code Starts HERE
    
    # load and process test data
    test_df = load_data_in_chunks(input_file)#, chunksize=10000)
    
    # load saved model
    bst = xgb.Booster()
    bst.load_model(model_file)
    
    # make prediction
    xg_test = xgb.DMatrix(test_df)
    c_prob = bst.predict(xg_test)
    
    # post process predicted probabilities
    c_time = test_df.reset_index()['time']
    c_suite = test_df.reset_index()['test_suite']
    scores = bst.get_score(importance_type='gain')
    c_ttf = sorted(scores, key=scores.get, reverse=True)[:3]
    
    # pass values into the post-process function
    full_pred = pd.DataFrame(columns=OUTPUT_HEADER)
    full_pred = test_postprocess(c_time, c_suite, c_prob, c_ttf, full_pred)
    
    # output the solution
    full_pred.to_csv(output_file, index=False)

    print('Testing finished.')

    return 0

if __name__ == "__main__":
    main()
