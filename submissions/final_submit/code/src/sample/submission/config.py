# category & headers
CATEGORY = {'low':0, 'medium':1, 'high':2, 'baseline':3, 'channelized':4, 'surprise':5}
REV_CATEGORY = { v:k for k,v in CATEGORY.items()}
OUTPUT_HEADER = ['timestamp', 'test_suite', 'predicted_induced_state',
       'three_sec_predicted_induced_state',
       'predicted_induced_state_confidence',
       'three_sec_predicted_induced_state_confidence', 'top_three_features']

PARAM = {
    'booster': 'gbtree',
    'objective': 'multi:softprob',  
    'num_class': 6,               
    'gamma': 0.1,                  
    'max_depth': 12,               
    'lambda': 2,                   
    'subsample': 0.7,              
    'colsample_bytree': 0.7,       
    'min_child_weight': 3,
#     'silent': 1,                   
    'eta': 0.007,                  # learning_rate
    'seed': 1000,
#     'nthread': 4,                  
}

# number of iteration for model training
NUM_ROUND = 2500
CHUNKSIZE = 500000