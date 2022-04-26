import re
import time
import gc
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def model_explore(trial_reg):
    '''
    Relevant Docstring
    '''
    #Collect the feature importance, sort feature importance and the select the top 2-22 features
    features = pd.DataFrame(index = trial_reg.feature_name(), data = trial_reg.feature_importance(importance_type='split'), columns = ['Feature_Weight'])
    features.sort_values(by = 'Feature_Weight', ascending = False, inplace=True)
    top100 = features.iloc(axis=0)[2:22]

    #Create df for feature importance
    top100.reset_index(inplace = True)
    top100.columns = ['Feature', 'Feature Weight']

    #Create seaborn plot
    sns.set_style("whitegrid")
    sns.set_context("paper")
    plot = sns.barplot(x="Feature Weight", y="Feature", palette = "mako", data=top100.sort_values(by="Feature Weight", ascending=False))

    return plot

if __name__ == '__main__':

    #Create relative path names
    import os
    dirname = os.path.dirname(__file__)

    # #Neptune.ai for monitoring
    # import neptune
    # neptune.init(project_qualified_name='clljos001/CCLDP',
    #             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMjZjZTlkNjEtZTU5ZS00MTYxLWFlNjYtNWQzNThlMDY1YmQzIn0=')

  
    # This Code is  used to import the omics datasets, feature engineer, and feature select.
    if not (Path.exists(Path('X.csv')) and Path.exists(Path('y.csv'))):
        #Load in data, perform basic data manipulation, feature selection and removal of erroneous data.
        import Ingest
        files_to_imp = 'E'
        features_to_imp = 500
        X, y = Ingest.Ingest(files_to_imp, features_to_imp)

    #Load in data
    print('Loading in Data')
    X = pd.read_csv(os.path.join(dirname,'X.csv'), index_col=0, engine="pyarrow")
    y = pd.read_csv(os.path.join(dirname, 'y.csv'), index_col=0, engine="pyarrow")
    
    print('convert categories')
    X.loc(axis=1)['PUTATIVE_TARGET'] = X.loc(axis=1)['PUTATIVE_TARGET'].astype('category')
    X.loc(axis=1)['DRUG_NAME'] = X.loc(axis=1)['DRUG_NAME'].astype('category')


    print('Convert drug names into numeric categories')
    X["DRUG_NAME_CAT"] = X["DRUG_NAME"].cat.codes
    X["PUTATIVE_TARGET_CAT"] = X["PUTATIVE_TARGET"].cat.codes

    lgbmX = X[X.columns.difference(['DRUG_NAME', 'PUTATIVE_TARGET'])]

    print('Creating Test & Train')
    #Create a test, validation and train set
    print("\nCreating test, train, and validation datasets")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(lgbmX, y, random_state = 12345, test_size=0.1, shuffle = True, stratify = X.loc(axis=1)['DRUG_NAME_CAT'])
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state = 12345, test_size=0.11, shuffle = True, stratify = X_train.loc(axis=1)['DRUG_NAME_CAT'])

    print('Creating LGBM Datasets')
    #LGB datasets are used for model training.
    import lightgbm as lgb
    train = lgb.Dataset(X_train, label = y_train, free_raw_data = False, categorical_feature=['DRUG_NAME_CAT', 'PUTATIVE_TARGET_CAT'])
    test = lgb.Dataset(X_test, label = y_test, free_raw_data = False, categorical_feature=['DRUG_NAME_CAT', 'PUTATIVE_TARGET_CAT'])
    val = lgb.Dataset(X_val, label = y_val, free_raw_data = False, categorical_feature=['DRUG_NAME_CAT', 'PUTATIVE_TARGET_CAT'])
    print("Complete")

    #Delete X and y to free up memory. This step isn't done in the code to cross validate.
    #del X, y

    #take out the trash to free up memory
    gc.collect()

    #### Train a single LightGBM Model####
    from sklearn.metrics import mean_squared_error as mse, r2_score

    #This dictionary is the parameter list for the model we will be training.
    p = {
        #'max_depth' : -1, #default = -1, no limit
        'max_bin' : 500, #default = 255
        #'num_leaves' : 15, #Default = 31 #XGBoost behav = 2^depth - 1
        'boosting' : 'gbdt', #Default = gdbt
        'metric' : 'rmse',
        'num_threads' : -1,
        'force_col_wise': True,
        #'use_missing' : False, #default = True
        'learning_rate' : 0.08, #default = 0.1
        #'feature_fraction' : 0.75,
        #'lambda_l1' : 0, #default = 0
        #'lambda_l2' : 0, #Default = 0
        'cat_smooth' : 150, #Default = 10
        'cat_l2': 300, #default = 10
        'min_data_per_group' :  150, #default = 100
        'max_cat_threshold' : 200, #default = 32
        'min_data_in_leaf' : 10, #default = 20
        #'extra_trees' : True, #default = False
        #'subsample' : 1.0, #Default = 1
        #'colsample_bytree' : 0.5,
        #'bagging_fraction' : 0.75, #Default = 1, used to deal with overfitting
        #'bagging_freq' : 100, #Default = 0, used to deal with overfitting in conjunction with bagging_fraction.
        #'path_smooth' : 150,
        }

    #This links to neptune.ai to monitor the training of the model.
    #Go to this link to the see the experiment: https://ui.neptune.ai/clljos001/CCLDP
    #neptune.create_experiment(name="CCL_DP", tags=['Desktop', 'Example_Code'],  params=p)

    print('Training Model')
    trial_reg = lgb.train(
                        params = p, #Parameter dictionary
                        train_set = train, #Train set created earlier
                        num_boost_round=20000, #How many rounds to train for unless early stopping is reached
                        valid_sets = [val], #Set to validate on
                        valid_names = ['Validation'],
                        early_stopping_rounds = 150, #If not performance improvements for 150 rounds then stop.
                        verbose_eval = 50, #Tell us how the training is doing every 50 rounds.
                        )

    #Save the model we just trained
    trial_reg.save_model('LightGBM_Model')

    ###Performance Metrics & Logging to Neptune.ai###
    test_pred = trial_reg.predict(X_test)
    r2_test = r2_score(y_test,test_pred)
    # neptune.log_metric('Test R2', r2_test)

    val_pred = trial_reg.predict(X_val)
    r2_val = r2_score(y_val,val_pred)
    # neptune.log_metric('Validation R2', r2_val)

    train_pred = trial_reg.predict(X_train)
    r2_train = r2_score(y_train, train_pred)
    # neptune.log_metric('Train R2', r2_train)

    print('R2 for Test: ', r2_test, '\nR2 for Validation: ', r2_val,  '\nR2 for Train: ', r2_train)

    rmse_test = math.sqrt(mse(y_test, test_pred))
    print("RMSE for Test: ", rmse_test)
    # neptune.log_metric('test rmse', rmse_test)

    rmse_train = math.sqrt(mse(y_train, train_pred))
    print("RMSE for Train: ", rmse_train)
    # neptune.log_metric('Test rmse', rmse_train)

    # Close the neptune uplink
    # neptune.stop()

    #Print the model's feature importance excluding drug_name and drug_target
    print(model_explore(trial_reg))

