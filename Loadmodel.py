import os
import math
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import mean_squared_error as mse, r2_score

#Load in regression model
regressor = lgb.Booster(model_file='LightGBM_Model')

#Create relative path names
dirname = os.path.dirname(__file__)

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


#Performance Metrics
test_pred = regressor.predict(X_test)
r2_test = r2_score(y_test,test_pred)

val_pred = regressor.predict(X_val)
r2_val = r2_score(y_val,val_pred)

train_pred = regressor.predict(X_train)
r2_train = r2_score(y_train, train_pred)

print('R2 for Test: ', r2_test, '\nR2 for Validation: ', r2_val,  '\nR2 for Train: ', r2_train)

rmse_test = math.sqrt(mse(y_test, test_pred))
print("RMSE for Test: ", rmse_test)

rmse_train = math.sqrt(mse(y_train, train_pred))
print("RMSE for Train: ", rmse_train)

#Save the predicted outputs.
output = pd.DataFrame(np.column_stack((test_pred, y_test.values[:, 0])), columns=['predictions', 'real'])
output.to_csv('lgbm_predictions_real.csv')
