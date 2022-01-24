import pandas as pd
import xgboost as xgb
import numpy as np
import datetime

from config import CATEGORY, REV_CATEGORY, OUTPUT_HEADER, CHUNKSIZE


def round_time(time_serie):
    time_serie = pd.to_numeric(time_serie)
    time_serie = time_serie.apply(lambda x: datetime.datetime.fromtimestamp(x / 1000000))
    time_serie = time_serie.dt.round('1s')
    time_serie = time_serie.apply(lambda x: int(datetime.datetime.timestamp(x) * 1000000))
    #     time_serie = time_serie.drop_duplicates()

    return time_serie


# return the mode(most comm) element in a series
def mode(series):
    return series.value_counts().index[0]


# return the mean value excluding -9999.9 the default value, if there's normal values
def normal_mean(series):
    # remove missing values
    series = pd.to_numeric(series)
    series.fillna(-9999.9, inplace=True)
    
    if series.nunique() > 1:
        return series[series > -9999.9].mean()
    return series.unique()

# preprocess raw data into model readable format
def data_preprocess(df, full_df):
    # map  'induced_state' for training data
    if ('induced_state' in df.columns):
        df["induced_state"] = df["induced_state"].replace(CATEGORY)

    # round time to whole seconds
    df['time'] = round_time(df['time'])

    # aggregate the data by 'time' & 'test_suite'
    col_merger = dict(zip(df.columns[2:], [normal_mean] * len(df.columns[2:])))
    if ('induced_state' in col_merger):
        col_merger['induced_state'] = mode
    df = df.groupby(['time', 'test_suite']).agg(col_merger)

    return full_df.append(df)

# loading data in chunks
def load_data_in_chunks(path, chunksize=CHUNKSIZE):
    df = pd.read_csv(path, chunksize=chunksize)

    # read the header cols
    with open(path) as f:
        header = f.readline().strip().split(',')

    # iterate thru the data in chunks and process them
    full_df = pd.DataFrame(columns=header).set_index(['time', 'test_suite'])
    for i, c in enumerate(df):
        full_df = data_preprocess(c, full_df)

        print(f'{(i + 1) * chunksize} rows loaded....')

    # futher aggregate the duplicates generated from different chunks
    col_merger = dict(zip(full_df.columns, [normal_mean] * len(full_df.columns)))
    if ('induced_state' in col_merger):
        col_merger['induced_state'] = mode
    full_df = full_df.groupby(['time', 'test_suite']).agg(col_merger)

    print('Data loaded.')

    return full_df

# preprocess steps for training data
def train_preprocess(df, test_size=0):
    # enforce dtypes
    df['induced_state'] = df['induced_state'].astype(int)
    df['tlx_score'] = df['tlx_score'].astype(int)

    drop_cols = ['induced_state']  # , 'tlx_score']
    X = df.loc[:, [c not in drop_cols for c in df.columns]]
    Y = df['induced_state']

    xg_train = xgb.DMatrix(X, label=Y)

    #     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=2)

    #     xg_train = xgb.DMatrix(X_train, label=y_train)
    #     xg_test = xgb.DMatrix( X_test, label=y_test)

    #     return xg_train, xg_test, X_train, X_test, y_train, y_test

    return xg_train


# post-process prediction result into submission format
def test_postprocess(c_time, c_suite, c_prob, c_ttf, full_pred):
    
    output_header = OUTPUT_HEADER
    
    # create a empty dataframe for chunk data
    c_df = pd.DataFrame(columns=output_header)
    
    # setting time & test_suite
    c_df[output_header[0]] = c_time
    c_df[output_header[1]] = c_suite
    
    # process predicted probabilties
    # trim prob into 3 decimal places
    c_prob = np.vectorize(lambda x: format(x, '.3f'))(c_prob).tolist()
    # find the pred(highest prob) index for each row 
    c_pred = [r.index(max(r)) for r in c_prob]
    # map the pred index into string instance
    c_pis = [REV_CATEGORY[p] for p in c_pred]
    
    c_tpis = c_pis
    c_tpis_prob = c_prob
    c_ttf = [c_ttf] * len(c_prob)

    c_df[output_header[2]] = c_pis
    c_df[output_header[3]] = c_tpis
    c_df[output_header[4]] = [str(r).replace(",", "").replace("\'", "") for r in c_prob]
    c_df[output_header[5]] = [str(r).replace(",", "").replace("\'", "") for r in c_tpis_prob]
    c_df[output_header[6]] = [str(r).replace(",", "") for r in c_ttf]
    
    return full_pred.append(c_df)