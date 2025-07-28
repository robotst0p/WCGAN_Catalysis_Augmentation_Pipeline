#change notes
#cingulate only data instead of rfe
#user model selection with optuna parameter tuning
#added in xgboost



#data processing libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sn

#normalization
from sklearn.preprocessing import StandardScaler

#model/training importing 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


#import LeaveOneOut for cross validation 
from sklearn.model_selection import LeaveOneOut

#metrics importing (accuracy, precision, sensitivity, recall)
from sklearn import metrics

#helping functions 
from helper_functions import retrieve_feature_names, feature_vote

#parameter tuning 
import optuna

#improve cpu training performance
from sklearnex import patch_sklearn
patch_sklearn()

#load in suvr data as pandas dataframe
raw_dataframe = pd.read_excel('C:/Users/meyer/Desktop/SUVr_Analysis/original_data/AUD_SUVR_wb_cingulate.xlsx', index_col = 0)

#map subject labels to numerical values 
raw_dataframe.loc[raw_dataframe["CLASS"] == "AUD", "CLASS"] = 1
raw_dataframe.loc[raw_dataframe["CLASS"] == "CONTROL", "CLASS"] = 0

processed_data = raw_dataframe

X_df = processed_data.drop(['CLASS'], axis = 1)

#convert to numpy array for training 
#X = X_df.to_numpy()
X = X_df
y = raw_dataframe['CLASS']
y = y.astype(int)

#z-score normalization
scaler = StandardScaler()

#raw_data normalization of feature vector
X_model = scaler.fit(X)
X_normal = X_model.transform(X)

#leaveoneout cross validation and decisiontree model creation 
loo = LeaveOneOut()
loo.get_n_splits(X)


from optuna.integration import OptunaSearchCV
from optuna.distributions import CategoricalDistribution, LogUniformDistribution

#prompt user to enter desired model type
print("MODELS:")
print("svm" + "\n" + "xgboost" + "\n" + "decisiontree" + "\n" + "randomforest") 

model_select = input("ENTER DESIRED MODEL OF THOSE LISTED: ")

if (model_select == "svm"):
    reset_model = svm.SVC(kernel='rbf', random_state=42)
elif (model_select == "randomforest"):
    reset_model = RandomForestClassifier()
elif (model_select == "xgboost"):
    reset_model = xgb.XGBClassifier()
elif (model_select == "decisiontree"):
    reset_model = DecisionTreeClassifier()

cv = LeaveOneOut()

#optuna search parameters for each model type
search_spaces_svm =  { "C": optuna.distributions.FloatDistribution(0.1, 10, step=0.1)
                }

search_spaces_logreg =  { "C": optuna.distributions.FloatDistribution(0.1, 10, step=0.1),
                         "penalty": optuna.distributions.CategoricalDistribution(["l1","l2","elasticnet"]),
                         "dual": optuna.distributions.CategoricalDistribution([True,False]),
                         "tol": optuna.distributions.FloatDistribution(0, .1, step = .1),
                         "random_state": optuna.distributions.IntDistribution(1, 100, step = 1),
                         "solver": optuna.distributions.CategoricalDistribution(["lbfgs","liblinear","newton-cg","newton-cholesky","sag","saga"])
                }

search_spaces_decisiontree =  { "criterion": optuna.distributions.CategoricalDistribution(["gini","entropy","log_loss"]),
                               "splitter": optuna.distributions.CategoricalDistribution(["best","random"]),
                               "max_depth": optuna.distributions.IntDistribution(1, 100, step = 1),
                               "min_samples_split": optuna.distributions.IntDistribution(2, 10, step = 1),
                               "random_state": optuna.distributions.IntDistribution(0, 100, step = 5)

                }

search_spaces_randomforest =  { "criterion": optuna.distributions.CategoricalDistribution(["gini","entropy"]),
                                "n_estimators": optuna.distributions.IntDistribution(100, 1000, step = 100),
                                "max_depth": optuna.distributions.IntDistribution(1, 100, step = 1),
                                "min_samples_split": optuna.distributions.IntDistribution(2, 10, step = 1)
                }
 
search_spaces_xgboost = {"tree_method": optuna.distributions.CategoricalDistribution(["auto","gpu_hist"])}
 
if (model_select == "svm"):
    param_select = search_spaces_svm
elif (model_select == "randomforest"):
    param_select = search_spaces_randomforest
elif (model_select == "xgboost"):
    param_select = search_spaces_xgboost
elif (model_select == "decisiontree"):
    param_select = search_spaces_decisiontree


optuna_search = OptunaSearchCV(
    estimator=reset_model,
    param_distributions = param_select,
    n_trials=10,
    cv=cv,
    error_score=0.0,
    refit=True,
)

optuna_search.fit(X_normal, y.astype(int))

optuna_search.best_score_
best_params = optuna_search.best_params_

if (model_select == "svm"):
    reset_model = svm.SVC(**best_params)
elif (model_select == "randomforest"):
    reset_model = RandomForestClassifier(**best_params)
elif (model_select == "xgboost"):
    reset_model = xgb.XGBClassifier(**best_params)
elif (model_select == "decisiontree"):
    reset_model = DecisionTreeClassifier(**best_params)

print(best_params) 

current_highest_score= optuna_search.best_score_

y_pred_list = []
y_test_list = []

for train_index, test_index in loo.split(X):
    mod_dt = reset_model

    #optuna_search = optuna.integration.OptunaSearchCV(estimator = mod_dt, param_distributions = search_spaces, n_trials = 10, cv = cv, error_score = 0.0, refit = True)

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #normalization of training and testing X
    train_normal = scaler.fit(X_train)
    X_train_normal = pd.DataFrame(train_normal.transform(X_train), columns = X_train.columns)
    X_test_normal = pd.DataFrame(train_normal.transform(X_test), columns = X_test.columns)

    y_test_list.append(y_test[0])

    #fit the model on the training data 
    mod_dt.fit(X_train_normal, y_train)
    #mod_dt.fit(X_train_normal, y_train)

    #predict the response for the test set
    y_pred = mod_dt.predict(X_test_normal)
    #y_pred = mod_dt.predict(X_test_normal)
    y_pred_list.append(y_pred)

print("Accuracy:", metrics.accuracy_score(y_test_list, y_pred_list))
print("Precision:", metrics.precision_score(y_test_list, y_pred_list))
print("Recall:", metrics.recall_score(y_test_list, y_pred_list))
print("METRICS REPORT:", metrics.classification_report(y_test_list, y_pred_list))