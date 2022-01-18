import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

def main():
    
    if len(sys.argv) < 2 or len(sys.argv[1]) == 0:
        print("Score Type is missing.")
        return 1
    
    if len(sys.argv) < 3 or len(sys.argv[2]) == 0:
        print("Ground Truth File is missing.")
        return 1
    
    if len(sys.argv) < 4 or len(sys.argv[3]) == 0:
        print("Submission File is missing.")
        return 1
    
    if len(sys.argv) < 5 or len(sys.argv[4]) == 0:
        print("Result File Path is missing.")
        return 1
    
    print('Scoring started.')

    ground_truth_data_file = './%s_data/%s' % (sys.argv[1], sys.argv[2])
    submission_data_file = sys.argv[3]
    result_path = sys.argv[4]

    ground_truth_data = pd.read_csv(ground_truth_data_file)
    submission_data = pd.read_csv(submission_data_file)

    try:
        category = {'low':0, 'medium':1, 'high':2, 'baseline':3, 'channelized':4, 'surprise':5}
        states = ['low', 'medium', 'high', 'baseline', 'channelized', 'surprise']
        features = ['E4_BVP', 'E4_GSR', 'LooxidLink_EEG_A3', 'LooxidLink_EEG_A4', 'LooxidLink_EEG_FP1', 'LooxidLink_EEG_FP2',
                    'LooxidLink_EEG_A7', 'LooxidLink_EEG_A8', 'Muse_EEG_TP9', 'Muse_EEG_AF7', 'Muse_EEG_AF8', 'Muse_EEG_TP10',
                    'Muse_PPG_0', 'Muse_PPG_1', 'Muse_PPG_2', 'Myo_GYR_X', 'Myo_GYR_Y', 'Myo_GYR_Z', 'Myo_EMG_0', 'Myo_EMG_1',
                    'Myo_EMG_2', 'Myo_EMG_3', 'Myo_EMG_4', 'Myo_EMG_5', 'Myo_EMG_6', 'Myo_EMG_7', 'PICARD_fnirs_0', 'PICARD_fnirs_1',
                    'Polar_bpm', 'Polar_hrv', 'ViveEye_eyeOpenness_L', 'ViveEye_pupilDiameter_L', 'ViveEye_pupilPos_L_X',
                    'ViveEye_pupilPos_L_Y', 'ViveEye_gazeOrigin_L_X', 'ViveEye_gazeOrigin_L_Y', 'ViveEye_gazeOrigin_L_Z',
                    'ViveEye_gazeDirection_L_X', 'ViveEye_gazeDirection_L_Y', 'ViveEye_gazeDirection_L_Z', 'ViveEye_eyeOpenness_R',
                    'ViveEye_pupilDiameter_R', 'ViveEye_pupilPos_R_X', 'ViveEye_pupilPos_R_Y', 'ViveEye_gazeOrigin_R_X',
                    'ViveEye_gazeOrigin_R_Y', 'ViveEye_gazeOrigin_R_Z', 'ViveEye_gazeDirection_R_X', 'ViveEye_gazeDirection_R_Y',
                    'ViveEye_gazeDirection_R_Z', 'Zephyr_HR', 'Zephyr_HRV']

        final_submission_data = submission_data.copy()
        final_submission_data = final_submission_data.drop_duplicates()
        final_submission_data = final_submission_data.reset_index(drop = True)
        final_submission_data = ((final_submission_data.groupby(['timestamp','test_suite'])).max()).reset_index()

        ttf = final_submission_data.top_three_features
        ttf_arr = ttf.apply(lambda x: x if isinstance(x, list) else [y for y in x.replace('[','').replace(']','').replace('\'','').split()])
        valid = all(ttf_arr.apply(lambda x: all(y in features for y in x)))
        if not valid:
            raise Exception("Invalid feature name in top_three_features")
        valid = all(ttf_arr.apply(lambda x: len(x) == 3))
        if not valid:
            raise Exception("Only top three features should be provided")
        
        pis = final_submission_data.predicted_induced_state
        pis_arr = pis.apply(lambda x: x if isinstance(x, list) else [y for y in x.replace('[','').replace(']','').replace('\'','').split()])
        valid = all(pis_arr.apply(lambda x: all(y in states for y in x)))
        if not valid:
            raise Exception("Invalid induced state in predicted_induced_state")
        
        tspis = final_submission_data.three_sec_predicted_induced_state
        tspis_arr = tspis.apply(lambda x: x if isinstance(x, list) else [y for y in x.replace('[','').replace(']','').replace('\'','').split()])
        valid = all(tspis_arr.apply(lambda x: all(y in states for y in x)))
        if not valid:
            raise Exception("Invalid induced state in three_sec_predicted_induced_state")

        ##############################
        #Getting AUC score 1 - For the predicted induced state
        ##############################

        data = ground_truth_data.copy()
        
        # creating instance of labelencoder
        labelencoder = LabelEncoder()
        # converting type of columns to 'category'
        data['induced_state'] = data['induced_state'].astype('category')
        # Assigning numerical values and storing in another column
        data["induced_state_Cat"]=data["induced_state"].replace(category)

        enc = OneHotEncoder(handle_unknown='ignore')
        # passing induced_state_Cat column (label encoded values of induced_state)
        enc_df = pd.DataFrame(enc.fit_transform(data[['induced_state_Cat']]).toarray())
        #Dropping the column induced_state_Cat which is no longer needed
        data.drop(columns=['induced_state_Cat'], inplace=True)
        # merge with main df induced_state on key values
        data = data.reset_index(drop=True)
        data = data.join(enc_df)

        #Need to make sure that at least one instance of each induced state is present in the ground truth file

        #Creating the dataframe for predicted induced state confidence values
        sco_confidence_list = final_submission_data.predicted_induced_state_confidence
        sco_arr = sco_confidence_list.apply(lambda x: x if isinstance(x, list) else [float(y) for y in x.replace('[','').replace(']','').split()])
        col0 = sco_arr.apply(lambda x: x[0] if len(x) > 0 else np.nan)
        col1 = sco_arr.apply(lambda x: x[1] if len(x) > 1 else np.nan)
        col2 = sco_arr.apply(lambda x: x[2] if len(x) > 2 else np.nan)
        col3 = sco_arr.apply(lambda x: x[3] if len(x) > 3 else np.nan)
        col4 = sco_arr.apply(lambda x: x[4] if len(x) > 4 else np.nan)
        col5 = sco_arr.apply(lambda x: x[5] if len(x) > 5 else np.nan)
        df_predict = pd.DataFrame({'low': col0, 'medium': col1, 'high': col2, 'baseline': col3, 'channelized': col4, 'surprise': col5})

        sub_data = final_submission_data.join(df_predict)

        final_data = pd.merge(data, sub_data, how="left", left_on=['timestamp','test_suite'], right_on=['timestamp','test_suite'])
        final_data['low'] = final_data.apply(lambda row: (1-row[0]) if np.isnan(row['low']) else row['low'], axis=1)
        final_data['medium'] = final_data.apply(lambda row: (1-row[1]) if np.isnan(row['medium']) else row['medium'], axis=1)
        final_data['high'] = final_data.apply(lambda row: (1-row[2]) if np.isnan(row['high']) else row['high'], axis=1)
        final_data['baseline'] = final_data.apply(lambda row: (1-row[3]) if np.isnan(row['baseline']) else row['baseline'], axis=1)
        final_data['channelized'] = final_data.apply(lambda row: (1-row[4]) if np.isnan(row['channelized']) else row['channelized'], axis=1)
        final_data['surprise'] = final_data.apply(lambda row: (1-row[5]) if np.isnan(row['surprise']) else row['surprise'], axis=1)

        #Calculating AUC values for all the 6 states separately
        
        fpr_0, tpr_0 , threshold_0 = roc_curve(data[0], final_data['low'])
        auc_0 = auc (fpr_0, tpr_0)

        fpr_1, tpr_1 , threshold_1 = roc_curve(data[1], final_data['medium'])
        auc_1 = auc (fpr_1, tpr_1)

        fpr_2, tpr_2 , threshold_2 = roc_curve(data[2], final_data['high'])
        auc_2 = auc (fpr_2, tpr_2)

        fpr_3, tpr_3 , threshold_3 = roc_curve(data[3], final_data['baseline'])
        auc_3 = auc (fpr_3, tpr_3)

        fpr_4, tpr_4 , threshold_4 = roc_curve(data[4], final_data['channelized'])
        auc_4 = auc (fpr_4, tpr_4)

        fpr_5, tpr_5 , threshold_5 = roc_curve(data[5], final_data['surprise'])
        auc_5 = auc (fpr_5, tpr_5)

        #calculating the final AUC score
        auc_final_1 = float((auc_0 + auc_1 + auc_2 + auc_3 + auc_4 + auc_5)/6)

        ##############################
        #Getting AUC score 2 - For the three second predicted induced state
        ##############################

        data = ground_truth_data.copy()

        # Ignoring the last 3 seconds in the session would be ignored for the above below
        data = data[pd.notnull(data.three_sec_induced_state)]

        # creating instance of labelencoder
        labelencoder = LabelEncoder()
        # converting type of columns to 'category'
        data['three_sec_induced_state'] = data['three_sec_induced_state'].astype('category')
        # Assigning numerical values and storing in another column
        data["three_sec_induced_state_Cat"]=data["three_sec_induced_state"].replace(category)

        enc = OneHotEncoder(handle_unknown='ignore')
        # passing induced_state_Cat column (label encoded values of induced_state)
        enc_df = pd.DataFrame(enc.fit_transform(data[['three_sec_induced_state_Cat']]).toarray())
        #Dropping the column induced_state_Cat which is no longer needed
        data.drop(columns=['three_sec_induced_state_Cat'], inplace=True)
        # merge with main df induced_state on key values after resetting the data df index
        data = data.reset_index(drop=True)
        data = data.join(enc_df)

        #Need to make sure that at least one instance of each induced state is present in the ground truth file

        #Creating the dataframe for predicted induced state confidence values
        sco_confidence_list = final_submission_data.three_sec_predicted_induced_state_confidence
        sco_arr = sco_confidence_list.apply(lambda x: x if isinstance(x, list) else [float(y) for y in x.replace('[','').replace(']','').split()])
        col0 = sco_arr.apply(lambda x: x[0] if len(x) > 0 else np.nan)
        col1 = sco_arr.apply(lambda x: x[1] if len(x) > 1 else np.nan)
        col2 = sco_arr.apply(lambda x: x[2] if len(x) > 2 else np.nan)
        col3 = sco_arr.apply(lambda x: x[3] if len(x) > 3 else np.nan)
        col4 = sco_arr.apply(lambda x: x[4] if len(x) > 4 else np.nan)
        col5 = sco_arr.apply(lambda x: x[5] if len(x) > 5 else np.nan)
        df_predict = pd.DataFrame({'low': col0, 'medium': col1, 'high': col2, 'baseline': col3, 'channelized': col4, 'surprise': col5})

        sub_data = final_submission_data.join(df_predict)

        final_data = pd.merge(data, sub_data, how="left", left_on=['timestamp','test_suite'], right_on=['timestamp','test_suite'])
        final_data['low'] = final_data.apply(lambda row: (1-row[0]) if np.isnan(row['low']) else row['low'], axis=1)
        final_data['medium'] = final_data.apply(lambda row: (1-row[1]) if np.isnan(row['medium']) else row['medium'], axis=1)
        final_data['high'] = final_data.apply(lambda row: (1-row[2]) if np.isnan(row['high']) else row['high'], axis=1)
        final_data['baseline'] = final_data.apply(lambda row: (1-row[3]) if np.isnan(row['baseline']) else row['baseline'], axis=1)
        final_data['channelized'] = final_data.apply(lambda row: (1-row[4]) if np.isnan(row['channelized']) else row['channelized'], axis=1)
        final_data['surprise'] = final_data.apply(lambda row: (1-row[5]) if np.isnan(row['surprise']) else row['surprise'], axis=1)

        #Calculating AUC values for all the 6 states separately

        fpr_0, tpr_0 , threshold_0 = roc_curve(data[0], final_data['low'])
        auc_0 = auc (fpr_0, tpr_0)

        fpr_1, tpr_1 , threshold_1 = roc_curve(data[1], final_data['medium'])
        auc_1 = auc (fpr_1, tpr_1)

        fpr_2, tpr_2 , threshold_2 = roc_curve(data[2], final_data['high'])
        auc_2 = auc (fpr_2, tpr_2)

        fpr_3, tpr_3 , threshold_3 = roc_curve(data[3], final_data['baseline'])
        auc_3 = auc (fpr_3, tpr_3)

        fpr_4, tpr_4 , threshold_4 = roc_curve(data[4], final_data['channelized'])
        auc_4 = auc (fpr_4, tpr_4)

        fpr_5, tpr_5 , threshold_5 = roc_curve(data[5], final_data['surprise'])
        auc_5 = auc (fpr_5, tpr_5)

        #calculating the final AUC score
        auc_final_2 = float((auc_0 + auc_1 + auc_2 + auc_3 + auc_4 + auc_5)/6)

        ##############################
        #Getting the final AUC score
        ##############################

        final_score = 70 * auc_final_1 + 30 * auc_final_2
        final_score = round(final_score, 5)

    except Exception as e:
        print(e)
        final_score = 0

    finally:
        print(final_score)
        with open(result_path + '/result.txt', 'w') as out:
            out.write(f'{final_score}\n')

    print('Scoring finished.')

    return final_score

if __name__ == "__main__":
    main()
