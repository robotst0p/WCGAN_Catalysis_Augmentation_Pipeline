import pandas as pd 
import numpy as np 

import tensorflow as tf
import pickle

from sklearn.preprocessing import StandardScaler

#pickle original normalized datasets (both control and subject) for synthetic comparison

raw_dataframe = pd.read_excel("C:/Users/meyer/Desktop/SUVr_Analysis/original_data/AUD_SUVR_wb_cingulate.xlsx", index_col=0)

#map subject labels to numerical values instead of categorical
raw_dataframe.loc[raw_dataframe["CLASS"] == "AUD", "CLASS"] = 1
raw_dataframe.loc[raw_dataframe["CLASS"] == "CONTROL", "CLASS"] = 0

processed_data = raw_dataframe

control_frame = processed_data.loc[(processed_data["CLASS"]) == 0]
aud_frame = processed_data.loc[(processed_data["CLASS"]) == 1]

control_frame_x = control_frame.drop(["CLASS"], axis = 1)
aud_frame_x = aud_frame.drop(["CLASS"], axis = 1)

X_df = processed_data.drop(["CLASS"], axis = 1)

# convert to numpy array for training
X = X_df.copy()
control_x = control_frame_x.copy()
aud_x = aud_frame_x.copy()

y = raw_dataframe["CLASS"]
y = y.astype(int)

# synth_y_container = pd.Series()
# synth_x_container = pd.DataFrame(columns=X.columns)


scaler1 = StandardScaler()
# raw_data normalization of feature vector
X_model = scaler1.fit(X)
control_x_model = scaler1.fit(control_x)
aud_x_model = scaler1.fit(aud_x)

X_normal = pd.DataFrame(X_model.transform(X), columns=X_df.columns)
control_frame_normal = pd.DataFrame(
    control_x_model.transform(control_x), columns=X_df.columns
)
aud_frame_normal = pd.DataFrame(aud_x_model.transform(aud_x), columns=X_df.columns)

aud_frame_normal.to_pickle("./aud_frame_normal.pkl")
control_frame_normal.to_pickle("./control_frame_normal.pkl")



