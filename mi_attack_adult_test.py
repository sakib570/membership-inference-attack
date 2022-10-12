#!/usr/bin/env python
# coding: utf-8

from mia_models import train_target_model, load_shadow_data, train_shadow_models
from mia_models import prepare_attack_test_data, prep_validation_data, prep_attack_train_data
from mia_models import shokri_attack, prety_print_result
import tensorflow.keras as keras
import numpy as np
from sklearn.utils import resample
import pandas as pd
import pickle
import os
import csv
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression
from statistics import mean
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.metrics import accuracy_score
MODEL_PATH = './model/'
DATA_PATH = './data/'



# --------------------------------------------Original Data--------------------------------------------------------------#

# original dataset with random index
dataset_name = 'adult'
train_size = 7000
attack_test_size = 2500
test_start = 7000
data = pd.read_csv('data/adult.data', na_values=["?"], header=None)
data.dropna(inplace=True)
target_dataset = data.sample(n = 10000, replace = False)
df_rest = data.loc[~data.index.isin(target_dataset.index)]
shadow_dataset = df_rest.sample(n = 12000, replace = False)
df_rest = df_rest.loc[~df_rest.index.isin(shadow_dataset.index)]
attack_test_nonmembers = df_rest.sample(n = attack_test_size, replace = False)
attack_test_members =  target_dataset.iloc[:train_size,:].sample(n = attack_test_size, replace = False)


# trian target model
per_class_sample=5000
channel=0   
epochs=200
act_layer=3
n_class = 2
is_synthetic = False
VERBOSE = 0
test_ratio = 0.3

target_model, dim = train_target_model(target_dataset, dataset_name, per_class_sample, epochs, act_layer, n_class, train_size, test_start, is_synthetic)

#train shadow model
n_shadow_models = 20
shadow_data_size = 10000

load_shadow_data(shadow_dataset, dataset_name, n_shadow_models, shadow_data_size, test_ratio, is_synthetic)
n_shadow_train_performance, n_shadow_test_performance, n_attack_data, x_shadow_train, y_shadow_train, x_shadow_test, y_shadow_test, shadow_model_init, shadow_accuracy = train_shadow_models(dataset_name, n_shadow_models, n_class, dim, epochs, channel)

#train attack model
attack_test_data = prepare_attack_test_data(dataset_name, attack_test_members, attack_test_nonmembers, target_model, is_synthetic)
mem_validation, nmem_validation = prep_validation_data(attack_test_data)
attack_train_df = prep_attack_train_data(n_attack_data)
pred_membership, ori_membership, TP_idx_list, TN_idx_list = shokri_attack(attack_train_df, mem_validation, nmem_validation, epochs)
tp, fp, fn, tn, precision, advj, acc, recall = prety_print_result (ori_membership,pred_membership)
print('Accuracy: ', acc, 'Precision: ', precision)

# --------------------------------------------Original Data--------------------------------------------------------------#