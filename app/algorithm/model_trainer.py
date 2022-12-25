#!/usr/bin/env python

import os, warnings, sys
import pprint
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
warnings.filterwarnings('ignore') 

import numpy as np, pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.cluster import KMeans

import algorithm.preprocessing.pipeline as pp_pipe
import algorithm.preprocessing.preprocess_utils as pp_utils
import algorithm.utils as utils
# import algorithm.scoring as scoring
from algorithm.model.clustering import ClusteringModel as Model, get_data_based_model_params
from algorithm.utils import get_model_config

from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score


# get model configuration parameters 
model_cfg = get_model_config()


def get_trained_model(train_data, data_schema, hyper_params):        
    
    # preprocess data
    print("Pre-processing data...")
    preprocessed_data, preprocess_pipe  = preprocess_data(train_data, data_schema)   
    train_X, ids = preprocessed_data['X'].astype(np.float), preprocessed_data['ids']
    # print('train_X shape:',  train_X.shape)  ; sys.exit()
                  
    # Get trained model  and clusters
    model, pred_clusters = train_model(train_X, hyper_params)    
    
    # return the prediction df with the id and prediction fields
    id_field_name = data_schema["inputDatasets"]["clusteringBaseMainInput"]["idField"] 
    preds_df = pd.DataFrame(ids, columns=[id_field_name])
    preds_df['prediction'] = pred_clusters
    # print(preds_df['prediction'].value_counts()) 
    
    return preprocess_pipe, model, preds_df


def train_model(train_X, hyper_params):
    
    # set random seeds
    utils.set_seeds()  
    
    # get model hyper-paameters parameters 
    data_based_params = get_data_based_model_params(train_X)
    model_params = { **hyper_params, **data_based_params }
    # print(model_params) 
    
    model = Model(  **model_params )  
    # train and get clusters
    pred_clusters = model.fit_predict(train_X)
    
    if len(set(pred_clusters)) == 1: score = np.float("inf")
    else: score = davies_bouldin_score(train_X, pred_clusters) 
    return model, pred_clusters
    


def preprocess_data(train_data, data_schema):
    # print('Preprocessing train_data of shape...', train_data.shape)
    pp_params = pp_utils.get_preprocess_params(train_data, data_schema, model_cfg) 
    # pprint.pprint(pp_params) 
    
    preprocess_pipe = pp_pipe.get_preprocess_pipeline(pp_params, model_cfg)
    train_data = preprocess_pipe.fit_transform(train_data)
    # print("Processed train X/y data shape", train_data['X'].shape, train_data['y'].shape)
          
    return train_data, preprocess_pipe 


def evaluate_clusters(X, clusters): 
    if len(set(clusters)) == 1: 
        score = np.float("inf")
    else: 
        score = davies_bouldin_score(X, clusters)        
        # score = calinski_harabasz_score(X, clusters)        
        # score = silhouette_score(X, clusters)   
    return score    
    