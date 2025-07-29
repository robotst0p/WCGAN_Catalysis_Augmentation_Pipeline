import matplotlib.pyplot as plt
import seaborn as sn

import pickle 
import pandas as pd
import numpy as np 

svm_cands_x = pd.read_pickle("..saved_data/final figures data/Figure 6 (Successful Candidates KDE Plot)/new_svm_cand_x.pkl")
svm_accuracy = pd.read_pickle("../saved_data/final figures data/Figure 5 (Model Metrics Plot)/new_svm_accuracy.pkl")
svm_sensitivity = pd.read_pickle("../saved_data/final figures data/Figure 5 (Model Metrics Plot)/new_svm_sensitivity.pkl")
svm_specificity = pd.read_pickle("../saved_data/final figures data/Figure 5 (Model Metrics Plot)/new_svm_specificity.pkl")

xgb_cands_x = pd.read_pickle("../saved_data/final figures data/Figure 6 (Successful Candidates KDE Plot)/new_xgb_cand_x.pkl")
xgb_accuracy = pd.read_pickle("../saved_data/final figures data/Figure 5 (Model Metrics Plot)/new_xgb_accuracy.pkl")
xgb_sensitivity = pd.read_pickle("../saved_data/final figures data/Figure 5 (Model Metrics Plot)/new_xgb_sensitivity.pkl")
xgb_specificity = pd.read_pickle("../saved_data/final figures data/Figure 5 (Model Metrics Plot)/new_xgb_specificity.pkl")

rf_cands_x = pd.read_pickle("../saved_data/new_rf_cand_x.pkl")
rf_accuracy = pd.read_pickle("../saved_data/new_rf_accuracy.pkl")
rf_sensitivity = pd.read_pickle("../saved_data/new_rf_sensitivity.pkl")
rf_specificity = pd.read_pickle("../saved_data/new_rf_specificity.pkl")

#svm green, xgboost blue, randomforest red
plt.plot(svm_x, svm_y, marker = "o", color = "green", label = "SVM", zorder = 1, linestyle = '--', linewidth = 3)  
plt.plot(rf_x, rf_y, marker = "o", color = "red", label = "RandomForest", linestyle = '--', zorder = 2, dashes = (5,5), linewidth = 3)
plt.plot(xgboost_x, xgboost_y, marker = "o", color = "blue", label = "XGBoost", zorder = 1, linestyle = '--', dashes = (5, 8), linewidth = 3)  
plt.legend(loc = 'upper left', fontsize = 40, frameon = False)
plt.xlabel("Synthetic Subjects Added to Dataset", fontsize = 50)
plt.ylabel("Model Specificity", fontsize = 50)

plt.xticks(fontsize = 35)
plt.yticks(fontsize = 35)

# plt.show()  
plt.clf()

#accuracy:
#svm: [0.7407407407407407, 0.7777777777777778, 0.8148148148148148, 0.8518518518518519, 0.8888888888888888]
#xgboost: [0.5925925925925926, 0.6296296296296297, 0.6666666666666666, 0.7037037037037037, 0.7407407407407407, 0.7777777777777778, 0.8148148148148148, 0.8518518518518519]
#randomforest: [0.5925925925925926, 0.6296296296296297, 0.6666666666666666, 0.7037037037037037, 0.7407407407407407, 0.7777777777777778, 0.8148148148148148, 0.8518518518518519]

#sensitivity
#svm: [0.3636, 0.5454545454545454, 0.5454545454545454, 0.6363636363636364, 0.7272727272727273, 0.8181818181818182]
#xgboost: [0.2727272727272727, 0.45454545454545453, 0.6363636363636364, 0.5454545454545454, 0.7272727272727273, 0.8181818181818182, 0.9090909090909091, 0.8181818181818182]
#randomforest: [0.2727272727272727, 0.45454545454545453, 0.6363636363636364, 0.6363636363636364, 0.5454545454545454, 0.7272727272727273, 0.9090909090909091, 1.0]

#specificity:
#svm: [0.875, 0.875, 0.9375, 0.9375, 0.9375, 0.9375]
#xgboost: [0.6875, 0.75, 0.6875, 0.8125, 0.75, 0.75, 0.75, 0.875]
#randomforest: [0.6875, 0.75, 0.6875, 0.75, 0.875, 0.8125, 0.75, 0.75]

