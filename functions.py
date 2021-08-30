# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 15:49:49 2021

@author: Alexandre
"""
import pickle
import pandas as pd
from scipy.special import expit 

def predict_modified(df, var_dict, sk_id, path):
    
    data = pd.DataFrame(df.loc[sk_id]).T
    data = data.drop('TARGET', axis=1)
    
    for key in var_dict.keys():
        data[key] = var_dict[key]
      
    data = data.T 
    X = data.values.reshape(1, -1)
    
    xgb = pickle.load(open(path + 'xgb_model.sav', 'rb'))
    probs = xgb.predict_proba(X)
    proba = probs[0][1]
    
    return proba, X
    
    

def transform_shap(proba, X, path):
    '''Cette fonction permet d'obtenir les shap values (valeurs d'influence
    chaque variable pour une prédiction donnée'''
    
    explainer = pickle.load(open(path + 'explainer.sav', 'rb'))
    shap_values = explainer.shap_values(X)[0]
    expected_value = explainer.expected_value
    model_prediction = proba

    shap_values_trans = shap_rescale(shap_values, expected_value, model_prediction)
    return shap_values_trans



def shap_rescale(shap_values, expected_value, model_prediction):  
    '''fonction permettant de remettre à l'échelle les shap values'''
    expected_value_transformed = expit(expected_value)  #fonction similaire à la régression logistique
    original_explanation_distance = sum(shap_values)    
    distance_to_explain = model_prediction - expected_value_transformed
    distance_coefficient = original_explanation_distance / distance_to_explain
    shap_values_transformed = shap_values / distance_coefficient
    
    return shap_values_transformed