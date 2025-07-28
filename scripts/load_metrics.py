import matplotlib.pyplot as plt
import seaborn as sn

import pickle 
import pandas as pd
import numpy as np 
raw_dataframe = pd.read_excel('C:/Users/meyer/Desktop/SUVr_Analysis/original_data/AUD_SUVR_wb_cingulate.xlsx') #,index_col = 0


raw_dataframe.loc[raw_dataframe["CLASS"] == "AUD", "CLASS"] = 1
raw_dataframe.loc[raw_dataframe["CLASS"] == "CONTROL", "CLASS"] = 0

y_train = raw_dataframe['CLASS']
    

# svm_accuracy_1 = pd.read_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/new_xgb_accuracy.pkl")
# svm_accuracy_2 = pd.read_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/xgb_accuracy.pkl")
# svm_accuracy_3 = pd.read_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/xg_accuracy.pkl")

# mcnemar_table = pd.read_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/new_stat_sig_svm_mcnemar_table.pkl")

# rf_parameters = pd.read_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/scripts/randomforest_hyperparameters.pickle")

print(y_train)


