#!/usr/bin/env python
# coding: utf-8
#
# DZNE Engagement - AML classifier
#
#

# ## 1. Imports

# In[1]:


# Importing Libraries

import pandas as pd
import numpy as np
import os
import os.path
import shutil
import time
import sys
import json
from datetime import datetime
import tarfile

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.regularizers import l1_l2
from keras.regularizers import l1
from keras.callbacks import EarlyStopping, ModelCheckpoint

# ## 2. Define constants

# In[2]:


# Declarae all the constants

# Experiment
exp_main = 'DZNE AML | ModelName: DZNE_AML_NN_10L | Dataset: {} | Scenario: {} | Node: {} | Experiment: {} | Permutation: {}-{}'
exp_main_abvr = 'dzne_{}_s{}_{}_{}_p{}_p{}'
modelName = 'DZNE_AML_NN_10L'

# CSV data file - defaults
datafile = "data1.csv"
infofile = "info1.csv"

# Customized split given by dzne
custom_trainset_file = "/data_{}_scen_{}/{}_train.txt"
custom_testset_file_node1 = "/data_{}_scen_{}/node_1_test.txt"
custom_testset_file_node2 = "/data_{}_scen_{}/node_2_test.txt"
custom_testset_file_node3 = "/data_{}_scen_{}/node_3_test.txt"

# Training and eval related constants
epochs = 100
batch_size = 64
num_nodes = 1024
dropout_rate = 0.1
l1_v = 0.0
l2_v = 0.005
binary_eval_threshold = 0.5

# Weights and eval file names
weightfile = '{}_model.h5'
model_eval_perm = '{}_result.txt'
model_eval_all = '{}_result.csv'


# ## 3. Obtaining the data

# In[3]:


def clean_data(data):
    data = data.rename(columns = {'Unnamed: 0':'Gene'})
    data = data.T
    data = data.rename(columns=data.iloc[0])
    data.index.names = ['Sample']
    data = data.iloc[1:]
    print('Data shape:', data.shape)
    return data

def clean_target(target):
    target = target.rename(columns = {'Unnamed: 0':'Sample'})
    target = target.set_index("Sample")
    print('Original target shape:', target.shape)
    actual_target = target[['Condition']]
    print('Actual target shape:', actual_target.shape)
    return actual_target

# Read complete XY data
def read_xy_data(dataDir, thisdatafile, thisinfofile):
    data = clean_data(pd.read_csv(dataDir + '/' + thisdatafile))
    target = clean_target(pd.read_csv(dataDir + '/' + thisinfofile))
    dt = data.join(target)
    dt['Condition_bin'] = dt['Condition'].map({'CASE':1, 'CONTROL':0})
    dt.Condition_bin.astype('int64')
    dt = dt.dropna()
    xy_data = dt
    print('XY data shape:', xy_data.shape)
    xy_data.head()
    print('CASE and CONTROL distribution in complete XY data - ')
    print(xy_data.Condition.value_counts())
    return xy_data


# ## 4. Partitioning or distributing the data

# In[4]:

def get_local_trainset_sample(dataDir, train_file, perm):
    df = pd.read_csv(dataDir + train_file, sep=' ', header=0)
    df_sample = df[[perm]].copy().dropna()
    print('Train sample shape = ', df_sample.shape)
    df_sample.drop_duplicates(subset=perm, keep='first', inplace=True)
    print('After dropping duplicates - Train sample shape = ', df_sample.shape)
    return df_sample


def extract_matching_dataset(df_dataset, matching_sample):
    df_dataset['sample_name'] = df_dataset.index.values
    matched_dataset = df_dataset[df_dataset.sample_name.isin(matching_sample)]
    matched_dataset = matched_dataset.drop(['sample_name'], axis=1)
    df_dataset = df_dataset.drop(['sample_name'], axis=1)
    print('Extracted matched dataset: [{},{}]'.format(matched_dataset.shape[0], matched_dataset.shape[1]))
    return matched_dataset


def get_total_testset_sample(dataDir, file_node1, file_node2, file_node3, perm):
    df1 = pd.read_csv(dataDir + file_node1, sep=' ', header=0)
    df1_sample = df1[[perm]].copy().dropna()
    df2 = pd.read_csv(dataDir + file_node2, sep=' ', header=0)
    df2_sample = df2[[perm]].copy().dropna()
    df3 = pd.read_csv(dataDir + file_node3, sep=' ', header=0)
    df3_sample = df3[[perm]].copy().dropna()
    print('Node1 test sample shape = ', df1_sample.shape)
    print('Node2 test sample shape = ', df2_sample.shape)
    print('Node3 test sample shape = ', df3_sample.shape)
    df_test_sample = df1_sample.append(df2_sample, ignore_index=True)
    df_test_sample = df_test_sample.append(df3_sample, ignore_index=True)
    print('Total test sample shape = ', df_test_sample.shape)
    df_test_sample.drop_duplicates(subset=perm, keep='first', inplace=True)
    print('After dropping duplicates - Total test sample shape = ', df_test_sample.shape)
    return df_test_sample

def show_train_test_data(df_train, df_test):
    # Print the details of final train and test data for this node
    result = 'Distribution: Train data >>>\n'
    result = result + 'Total = ' + str(df_train.shape[0]) + '\n'
    result = result + str(df_train['Condition'].value_counts()) + '\n'
    result = result + 'Distribution: Test data >>>\n'
    result = result + 'Total = ' + str(df_test.shape[0]) + '\n'
    result = result + str(df_test['Condition'].value_counts()) + '\n'
    print(result)
    return result

# ## 5. Normalized the data

# In[5]:


def get_xy(xy_df):
    x_df = xy_df.drop(columns = ['Condition','Condition_bin'])
    y_df = xy_df[['Condition_bin']]
    print(x_df.shape)
    print(y_df.shape)
    return x_df, y_df

def get_norm_xy(train_test_df):
    scaler = StandardScaler()
    X_df, y_df = get_xy(train_test_df)
    X_df_norm = pd.DataFrame(scaler.fit_transform(X_df), columns=X_df.columns)
    return X_df_norm, y_df

def get_normalized_xy(df_train, df_test):
    X_train, y_train = get_norm_xy(df_train)
    X_test, y_test = get_norm_xy(df_test)
    return X_train, y_train, X_test, y_test


# ## 6. Create LR model for training

# In[6]:


# Create model
# Simple Lasso LR model
def create_model(r,c,lam = 0.01):
    # Note : r (rows) is not used
    # model is linear 
    # ypred = W1*i1 + W2*i2 + ... Wn*in + b 
    # A single unit dense layer, with linear activation will do.
    # for LASSO LR, idea is to compute cost as 
    # C = Reduced_Sum_of_Squares(W) + Lamda * L1(W) 
    # however keras does not have RSS function instead it has 
    # mean_Squared_error which is 1/N * RSS(W) , N: is batch size
    # this is a fairly good approximation of a Textbook LASSO LR 
    # C = mean_Squared_error(W) + Lamda * L1(W)
    model = Sequential()
    model.add(Dense(1, activation='linear', kernel_regularizer = l1(lam), input_dim=c))
    model.add(BatchNormalization())
    opt = keras.optimizers.SGD(learning_rate=0.01)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
    model.summary()
    return model

# ## 7. Evaluate the model

# In[7]:


def get_curr_time():
    now = datetime.now()
    curr_time_str = now.strftime("%d/%m/%Y %H:%M:%S")
    return curr_time_str


def write_exp_title(filepath, exp_name_main=None, exp_name_perm=None, data_dist=None, summary=False):
    resultfile = open(filepath, "a+")
    if exp_name_main != None:
        if summary == False:
            resultfile.write('==============================================================================================================================================\n')
            resultfile.write('Experiment Name - %s\n' % (exp_name_main))
            resultfile.write('Experiment Time - %s\n' % (get_curr_time()))
            resultfile.write('==============================================================================================================================================\n')
            if data_dist != None:
                resultfile.write('\n' + data_dist + '\n')
        else:
            resultfile.write('Permutation,Loss,Accuracy,ConfusionMatrix,Precision_1,Recall_1,F1-Score_1,Support_1,Precision_0,Recall_0,F1-Score_0,Support_0\n')
    if exp_name_perm != None:
        if summary == False:
            resultfile.write('\n==========================\n')
            resultfile.write('Permutation Name - %s\n' % (exp_name_perm))
            resultfile.write('===========================\n')
        else:
            resultfile.write('%s,' % (exp_name_perm))
    resultfile.flush()
    resultfile.close()

def write_result(filepath, result):
    resultfile = open(filepath, "a+")
    resultfile.write(result)
    resultfile.flush()
    resultfile.close()

# Evaluating the score the of the model against unseen data
def show_evaluation(filepath, model, X_test, y_test):
    score = model.evaluate(X_test, y_test, verbose = 1)
    result = 'Evaluation result ==>\n'
    result = result + 'Test loss: {}\n'.format(score[0])
    result = result + 'Test accuracy: {}\n'.format(score[1])
    print(result)
    write_result(filepath, result)
    summary_result = "{:0.2f},{:0.2f},".format(score[0],score[1])
    return summary_result

# Binary prediction
def show_confusion_matrix(filepath, model, X_test, y_test):
    y_pred_num = model.predict(X_test)
    y_pred = np.where(y_pred_num > binary_eval_threshold, 1, 0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    result = 'Confusion matrix result ==>\n'
    result = result + '{}\n'.format(conf_matrix)
    result = result + 'Model Output - Confusion Matrix: tp=%d, tn=%d, fp=%d, fn=%d\n' % (tp, tn, fp, fn)
    print(result)
    write_result(filepath, result)
    summary_result = 'tp=%d tn=%d fp=%d fn=%d,' % (tp, tn, fp, fn)
    return summary_result

# Performance - Classification report
def show_classification_report(filepath, model, X_test, y_test):
    y_pred_num = model.predict(X_test)
    y_pred = np.where(y_pred_num > binary_eval_threshold, 1, 0)
    class_report = classification_report(y_test, y_pred)
    result = 'Classification report result ==>\n'
    result = result + '{}\n'.format(class_report)
    print(result)
    write_result(filepath, result)
    class_report_dict = classification_report(y_test, y_pred, output_dict=True)
    label_1_dict = class_report_dict["1"]
    label_0_dict = class_report_dict["0"]
    label_1_summary = "{:0.2f},{:0.2f},{:0.2f},{}".format(label_1_dict["precision"], label_1_dict["recall"], label_1_dict["f1-score"], label_1_dict["support"])
    label_0_summary = "{:0.2f},{:0.2f},{:0.2f},{}".format(label_0_dict["precision"], label_0_dict["recall"], label_0_dict["f1-score"], label_0_dict["support"])
    summary_result = "{},{}\n".format(label_1_summary, label_0_summary)
    return summary_result

def write_y_result_in_csv(filepath, model, X_test, y_test):
    y_pred_num = model.predict(X_test)
    y_pred = np.where(y_pred_num > binary_eval_threshold, 1, 0)
    y_result_df = pd.DataFrame(y_test)
    y_result_df.rename(columns={'Condition_bin':'Actuals'}, inplace=True)
    y_result_df['Predictions'] = np.squeeze(y_pred)
    y_result_df['Actuals'] = y_result_df['Actuals'].map(lambda val: 'CASE' if val == 1 else 'CONTROL')
    y_result_df['Predictions'] = y_result_df['Predictions'].map(lambda val: 'CASE' if val == 1 else 'CONTROL')
    y_result_df.to_csv(filepath, index=True)


# ## 8. Main method - Entry point for Swarm

# In[8]:

def main(dataDir, xy_data, perm, base_out_dir,lasso_lamda):
    # Define permutation experiment
    print('\n==================================')
    print('>>> Permutation name: %s' % (perm))
    print('==================================\n')

    # Create perm sub dirs for storing results
    outDir = os.path.join(base_out_dir, perm)
    os.makedirs(outDir)

    # Use customized train / test set for distribution
    print('Using custom train and test set ...')
    df_train_sample = get_local_trainset_sample(dataDir, custom_trainset_file, perm)
    this_df_train = extract_matching_dataset(xy_data, df_train_sample[perm])

    df_testset_sample = get_total_testset_sample(dataDir, custom_testset_file_node1, custom_testset_file_node2, custom_testset_file_node3, perm)
    this_df_test = extract_matching_dataset(xy_data, df_testset_sample[perm])
    data_dist = show_train_test_data(this_df_train, this_df_test) # print details

    # Get normalized train and test data in X and y form
    X_train, y_train, X_test, y_test = get_normalized_xy(this_df_train, this_df_test)

    # Get the model
    rows = X_train.shape[0]  # number of rows
    columns = X_train.shape[1] # number of columns
    model = create_model(rows,columns,float(lasso_lamda))

    # Create callbacks
    # Early stopping callback if the loss function decreases
    esCallback = EarlyStopping(monitor='loss', min_delta=0.01, patience=20, verbose=2, mode='min')
    callbacks = [esCallback]

    # Starting training the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    # Save final model
    model.save(outDir + '/' + weightfile.format(perm))

    # Performance evaluation
    filepath = outDir + '/' + model_eval_perm.format(perm)
    write_exp_title(filepath, exp_name_main=exp_main, exp_name_perm=perm, data_dist=data_dist, summary=False)
    eval_result = show_evaluation(filepath, model, X_test, y_test)
    cm_result = show_confusion_matrix(filepath, model, X_test, y_test)
    cr_result = show_classification_report(filepath, model, X_test, y_test)
    predfile = outDir + '/' + perm + '_act_vs_pred.csv'
    write_y_result_in_csv(predfile, model, X_test, y_test)

    print('>>> Experiment done!\n')
    return eval_result, cm_result, cr_result


# Default configuration
def get_default_config():
    dataset = 'ds1'
    scenario = '1a'
    node = 'node_2'
    start_perm_index = 1
    stop_perm_index = 3
    return dataset, scenario, node, start_perm_index, stop_perm_index


# Get config from json file
def get_config(conf_file):
    with open(conf_file) as f:
        conf = json.load(f)
    return conf['dataset'], conf['scenario'], conf['node'], int(conf['start_perm_index']), int(conf['stop_perm_index'])


def make_tar_gz(output_file, source_dir):
    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

# ## 9. Entry point for python script

# In[9]:

if __name__ == '__main__':
    # Read environment variables if any
    data_dir = os.getenv('DATA_DIR', '../data')
    model_dir = os.getenv('MODEL_DIR', './')
    sentinel_ip = os.getenv('SENTINEL_NODE_IP', '127.0.0.1')
    this_ip = os.getenv('THIS_NODE_IP', '127.0.0.1')
    # lambda for the L1 weight regularizer 
    lasso_lamda = os.getenv('LAM', '0.01')

    # Get the config
    config_file = os.path.join(model_dir, 'modelconfig.json')
    if os.path.exists(config_file):
        print('Reading confuration from JSON config file ...')
        dataset, scenario, node, start_perm_index, stop_perm_index = get_config(config_file)
    else:
        print('No config file. Reading default configuration ...')
        dataset, scenario, node, start_perm_index, stop_perm_index = get_default_config()

    # Initialize depending on configuration
    exp_type = 'sn'

    exp_main = exp_main.format(dataset, scenario, node, exp_type, start_perm_index, stop_perm_index)
    exp_main_abvr = exp_main_abvr.format(dataset, scenario, node, exp_type, start_perm_index, stop_perm_index)
    custom_trainset_file = custom_trainset_file.format(dataset, scenario, node)
    custom_testset_file_node1 = custom_testset_file_node1.format(dataset, scenario)
    custom_testset_file_node2 = custom_testset_file_node2.format(dataset, scenario)
    custom_testset_file_node3 = custom_testset_file_node3.format(dataset, scenario)
    if dataset == 'ds1':
        datafile = "data1.csv"
        infofile = "info1.csv"
        batch_size = 64
    elif dataset == 'ds2':
        datafile = "data2.csv"
        infofile = "info2.csv"
        batch_size = 128
    elif dataset == 'ds3':
        datafile = "data3.csv"
        infofile = "info3.csv"
        batch_size = 32
    else:
        sys.exit("Dataset is not supported")

    # Start experiment
    print('\n*** Starting experiment: %s\n' % (exp_main))

    # Read data tables one time
    xy_data = read_xy_data(data_dir, datafile, infofile)

    # Create experiment directory
    base_out_dir_name = exp_main_abvr
    model_eval_all = model_eval_all.format(base_out_dir_name)
    base_out_dir = os.path.join(model_dir, base_out_dir_name)
    if os.path.exists(base_out_dir):
        shutil.rmtree(base_out_dir)
    os.makedirs(base_out_dir)
    print('Successfully created output directory - ', base_out_dir_name)

    # Loop over each permutation and train+evaluate
    filepath = base_out_dir + '/' + model_eval_all
    write_exp_title(filepath, exp_name_main=exp_main, summary=True)
    for index in range(start_perm_index, stop_perm_index+1):
        perm = 'perm_' + str(index)
        eval_result, cm_result, cr_result = main(data_dir, xy_data, perm, base_out_dir,lasso_lamda)
        write_exp_title(filepath, exp_name_perm=perm, summary=True)
        write_result(filepath, eval_result)
        write_result(filepath, cm_result)
        write_result(filepath, cr_result)

    # Create compressed tar of the output dir and delete output dir
    output_tarfile = os.path.join(model_dir, base_out_dir_name + '.tar.gz')
    make_tar_gz(output_tarfile, base_out_dir)
    # shutil.rmtree(base_out_dir)
    print('Successfully created the tar file with all results - %s' % (os.path.basename(output_tarfile)))

    print('\n*** Finished experiment: %s ***\n' % (exp_main))

