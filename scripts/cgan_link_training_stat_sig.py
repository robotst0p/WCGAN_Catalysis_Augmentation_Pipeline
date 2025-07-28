import pandas as pd
import numpy as np

import time
import pickle
import warnings

warnings.filterwarnings("ignore")
# normalization
from sklearn.preprocessing import StandardScaler

# model/training importing
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut
from sklearn import metrics

import xgboost as xgb

# parameter tuning
import optuna
from optuna.integration import OptunaSearchCV
from optuna.distributions import CategoricalDistribution, LogUniformDistribution

# tensorflow for cgan model loading
from tensorflow.keras.models import load_model

# import generator model
from lib import gan_architecture as gan

# load trained cgan
generator = load_model(
    "C:/Users/meyer/Desktop/SUVr_Analysis/scripts/old_weights/wgan_CingulateSUVR_29999.h5"
)
synthetic_suvr = gan.test_generator(generator)


# load in suvr data or only cingulate data as pandas dataframe
raw_dataframe = pd.read_excel("C:/Users/meyer/Desktop/SUVr_Analysis/original_data/AUD_SUVR_wb_cingulate.xlsx", index_col=0)

# map subject labels to numerical values
raw_dataframe.loc[raw_dataframe["CLASS"] == "AUD", "CLASS"] = 1
raw_dataframe.loc[raw_dataframe["CLASS"] == "CONTROL", "CLASS"] = 0

# keep original raw data intact in case we want to use later
processed_data = raw_dataframe

control_frame = processed_data.loc[(processed_data["CLASS"]) == 0]
aud_frame = processed_data.loc[(processed_data["CLASS"]) == 1]

control_frame_x = control_frame.drop(["CLASS"], axis=1)
aud_frame_x = aud_frame.drop(["CLASS"], axis=1)

X_df = processed_data.drop(["CLASS"], axis=1)

# convert to numpy array for training
X = X_df.copy()

y = raw_dataframe["CLASS"]
y = y.astype(int)

synth_y_container = pd.Series()
synth_x_container = pd.DataFrame(columns=X.columns)

scaler1 = StandardScaler()
# raw_data normalization of feature vector
X_model = scaler1.fit(X)
X_normal = pd.DataFrame(X_model.transform(X), columns=X_df.columns)

# prompt user to enter desired model type
print("MODELS:")
print("svm" + "\n" + "xgboost" + "\n" + "decisiontree" + "\n" + "randomforest")

model_select = input("ENTER DESIRED MODEL OF THOSE LISTED: ")

if model_select == "svm":
    reset_model = svm.SVC(kernel="rbf", random_state=42)
elif model_select == "randomforest":
    reset_model = RandomForestClassifier()
elif model_select == "xgboost":
    reset_model = xgb.XGBClassifier()
elif model_select == "decisiontree":
    reset_model = DecisionTreeClassifier()

# optuna search parameters for each model type -> these are the hyperparameters that will be tested for each model to determine 
# which ones are the best
search_spaces_svm = {"C": optuna.distributions.FloatDistribution(0.1, 10, step=0.1)}

search_spaces_logreg = {
    "C": optuna.distributions.FloatDistribution(0.1, 10, step=0.1),
    "penalty": optuna.distributions.CategoricalDistribution(["l1", "l2", "elasticnet"]),
    "dual": optuna.distributions.CategoricalDistribution([True, False]),
    "tol": optuna.distributions.FloatDistribution(0, 0.1, step=0.1),
    "random_state": optuna.distributions.IntDistribution(1, 100, step=1),
    "solver": optuna.distributions.CategoricalDistribution(
        ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
    ),
}

search_spaces_decisiontree = {
    "criterion": optuna.distributions.CategoricalDistribution(
        ["gini", "entropy", "log_loss"]
    ),
    "splitter": optuna.distributions.CategoricalDistribution(["best", "random"]),
    "max_depth": optuna.distributions.IntDistribution(1, 100, step=1),
    "min_samples_split": optuna.distributions.IntDistribution(2, 10, step=1),
    "random_state": optuna.distributions.IntDistribution(0, 100, step=5),
}

search_spaces_randomforest = {
    "criterion": optuna.distributions.CategoricalDistribution(["gini", "entropy"]),
    "criterion": optuna.distributions.CategoricalDistribution(["entropy", "gini"]),
    "n_estimators": optuna.distributions.IntDistribution(100, 1000, step=100),
    "max_depth": optuna.distributions.IntDistribution(1, 100, step=1),
  
    "min_samples_split": optuna.distributions.IntDistribution(2, 10, step=1)

}

search_spaces_xgboost = {
    "tree_method": optuna.distributions.CategoricalDistribution(["auto", "gpu_hist"])
}

if model_select == "svm":
    param_select = search_spaces_svm
elif model_select == "randomforest":
    param_select = search_spaces_randomforest
elif model_select == "xgboost":
    param_select = search_spaces_xgboost
elif model_select == "decisiontree":
    param_select = search_spaces_decisiontree

cv = LeaveOneOut()

optuna_search = OptunaSearchCV(
    estimator=reset_model,
    param_distributions=param_select,
    n_trials=20,
    cv=cv,
    error_score=0.0,
    refit=True,
)

optuna_search.fit(X_normal, y.astype(int))

best_model = optuna_search.best_estimator_

optuna_search.best_score_
best_params = optuna_search.best_params_

if model_select == "svm":
    reset_model = svm.SVC(**best_params)
    with open('svm_statsig_hyperparameters_test.pkl', 'wb') as handle:
        pickle.dump(best_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

elif model_select == "randomforest":
    reset_model = RandomForestClassifier(**best_params)
    with open('randomforest_statsig_hyperparameters.pickle', 'wb') as handle:
        pickle.dump(best_params, handle, protocol = pickle.HIGHEST_PROTOCOL)

elif model_select == "xgboost":
    reset_model = xgb.XGBClassifier()
    with open('xgboost_statsig_hyperparameters.pkl', 'wb') as handle:
        pickle.dump(best_params, handle, protocol = pickle.HIGHEST_PROTOCOL)

elif model_select == "decisiontree":
    reset_model = DecisionTreeClassifier(**best_params)
    with open('decisiontree_hyperparameters.pkl', 'wb') as handle:
        pickle.dump(best_params, handle, protocol = pickle.HIGHEST_PROTOCOL)

print(best_params)

#manually test the unaugmented model (0 synthetic subjects) for initial sensitivity and specificity metrics
y_true = []
y_pred = []

#leave one out cross validation
loo = LeaveOneOut()
loo.get_n_splits(X)

for train_index, test_index in loo.split(X_normal):
    X_train, X_test = X_normal.iloc[train_index], X_normal.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    best_model.fit(X_train, y_train)
    pred = best_model.predict(X_test)

    y_true.append(y_test[0])
    y_pred.append(pred[0])


tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

del y_true
del y_pred

current_highest_score = optuna_search.best_score_

succesful_cand_X = pd.DataFrame()  
succesful_cand_Y = pd.Series()

synth_counter = 0

accuracy_list = []
sensitivity_list = []
specificity_list = []

y_pred_real_list = []
y_pred_augmented_list = []
running_ground_truth = []

training_iterations = 0

#add first accuracy with no synthetic candidates to accuracy list before training loop begins
sensitivity_list.append(sensitivity)
specificity_list.append(specificity)
accuracy_list.append(current_highest_score)

while synth_counter <= 27 and training_iterations <= 100:
    print(training_iterations)
    
    synthetic_suvr = gan.test_generator(generator)
    synth_frame_x = pd.DataFrame(data=synthetic_suvr[0], columns=X_df.columns)
    synth_frame_y = pd.Series(synthetic_suvr[1])

    scaler2 = StandardScaler()

    #normalization of synthetic data
    X_model2 = scaler2.fit(synth_frame_x)
    synth_X_normal = pd.DataFrame(
        X_model2.transform(synth_frame_x), columns=X_df.columns
    )

    for row in list(synth_X_normal.index.values):
        y_pred_list = []
        y_test_list = []

        svc_real = clone(reset_model)
        svc = clone(reset_model)

        for train_index, test_index in loo.split(X_normal):

            X_train, X_test = X_normal.iloc[train_index], X_normal.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            #append synthetic point to X_training set
            synth_cand_x = synth_X_normal.loc[row]

            X_train_intermediate = X_train.append(
                synth_cand_x, ignore_index=True
            )  #xtrain intermediate is a orig+synthetic data

            #append synthetic point to y_training set
            synth_train_y = synth_frame_y.loc[row]

            y_train_intermediate = y_train.copy()

            y_train_intermediate.at[len(y_train) + 1] = synth_train_y

            y_train_intermediate = y_train_intermediate.rename(
                {len(y_train_intermediate): row}
            )

            X_train_intermediate = X_train_intermediate.append(succesful_cand_X)
            y_train_intermediate = y_train_intermediate.append(succesful_cand_Y)

            svc_real.fit(X_train, y_train)
            svc.fit(X_train_intermediate, y_train_intermediate)

            y_pred = svc.predict(X_test)
            y_pred_real = svc_real.predict(X_test)

            #collect predictions from model trained on real data and model trained on synthetic + real data
            y_pred_real_list.append(y_pred_real[0])
            y_pred_augmented_list.append(y_pred[0])

            y_test_list.append(y_test[0])
            y_pred_list.append(y_pred[0])

            running_ground_truth.append(y_test[0])

        y_pred_final = pd.Series(y_pred_list)

        del svc
        del svc_real
        
        score = metrics.accuracy_score(y, y_pred_final)

        #find highest f1 score to compare new f1 score to
        if score > current_highest_score:
            current_highest_score = score
            
            #calculate specificity and sensitivity using true negatives, false positives, false negatives and true positives given by the confusion matrix
            tn, fp, fn, tp = metrics.confusion_matrix(y_test_list, y_pred_list).ravel()
            specificity = tn / (tn + fp)
            sensitivity = tp / (tp + fn)

            print("SPECIFICITY: " + str(specificity))
            print("SENSITIVITY: " + str(sensitivity))

            model_precision = metrics.precision_score(y_test_list, y_pred_list)
            model_recall = metrics.recall_score(y_test_list, y_pred_list)

            accuracy_list.append(score)
            specificity_list.append(specificity)
            sensitivity_list.append(sensitivity)
            
            succesful_cand_X = succesful_cand_X.append(synth_cand_x)
            
            succesful_cand_Y = succesful_cand_Y.append(y_train_intermediate[-1:])
            succesful_cand_X.to_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/new_stat_sig_rf_cand_x_test.pkl")
            succesful_cand_Y.to_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/new_stat_sig_rf_cand_y_test.pkl")

            with open("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/new_stat_sig_rf_accuracy_test.pkl", 'wb') as f:
                pickle.dump(accuracy_list, f)
            
            with open("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/new_stat_sig_rf_sensitivity_test.pkl", 'wb') as f:
                pickle.dump(sensitivity_list, f)

            with open("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/new_stat_sig_rf_specificity_test.pkl", 'wb') as f:
                pickle.dump(specificity_list, f)

            print("ACCURACY INCREASED, SYNTHETIC CANDIDATE ADDED")
            print("NEW ACCURACY: " + str(score))
            print(synth_cand_x)

            print("Accuracy:", metrics.accuracy_score(y_test_list, y_pred_list))
            print("Precision:", model_precision)
            print("Recall:", model_recall)

            print(
                "METRICS REPORT:",
                metrics.classification_report(y_test_list, y_pred_list),
            )

            time.sleep(1)

            synth_counter += 1
            

    training_iterations += 1

#statistical significance test (McMahen)
from statsmodels.stats.contingency_tables import mcnemar 
import numpy as np

y_true = np.array(running_ground_truth)
y_pred_real = np.array(y_pred_real_list)
y_pred_augmented = np.array(y_pred_augmented_list)

#compare prediction correctness 
correct_real = (y_pred_real == y_true).astype(int)
correct_aug = (y_pred_augmented == y_true).astype(int)

#contingency table 
n_01 = np.sum((correct_real == 1) & (correct_aug == 0))
n_10 = np.sum((correct_real == 0) & (correct_aug == 1))

table = [[0, n_01],
            [n_10, 0]]

result = mcnemar(table, exact = False, correction = True)

print(table)

print("McNemar's test results:")
print("  Statistic =", result.statistic)
print("  p-value   =", result.pvalue)


#save stats from test
with open("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/new_stat_sig_rf_mcnemar_pred_real.pkl", 'wb') as f:
    pickle.dump(y_pred_real, f)

with open("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/new_stat_sig_rf_mcnemar_pred_augment.pkl", 'wb') as f:
    pickle.dump(y_pred_augmented, f)

with open("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/new_stat_sig_rf_mcnemar_ground_truth.pkl", 'wb') as f:
    pickle.dump(y_true, f)

with open("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/new_stat_sig_rf_mcnemar_table.pkl", 'wb') as f:
    pickle.dump(table, f)

with open("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/new_stat_sig_rf_mcnemar_p_value.pkl", 'wb') as f:
    pickle.dump(result.pvalue, f)

with open("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/new_stat_sig_rf_mcnemar_statistic.pkl", 'wb') as f:
    pickle.dump(result.statistic, f)
            
if result.pvalue < 0.05:
    print("Statistically significant improvement (p < 0.05)")
else:
    print("No statistically significant improvement")