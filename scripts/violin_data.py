import pandas as pd
import pickle
import pandas as pd
import pickle
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from lib import gan_architecture as gan
from sklearn.preprocessing import StandardScaler
from scipy.special import rel_entr
from scipy.stats import entropy

def normalize_frame(frame):
    scaler2 = StandardScaler()

    synth_model = scaler2.fit(frame)
    synth_frame_normal = pd.DataFrame(
        synth_model.transform(frame), columns=frame.columns
    )

    return synth_frame_normal

# load original dataframe as reference
aud_normal_x = pd.read_pickle("../saved_data/aud_frame_normal.pkl")

# generate 100 sets of synthetic samples, average them, add them to one frame until desired count is reached, then normalize and analyze
# normalize them at the end
def add_samples(old_frame, type, iteration, new_frame=pd.DataFrame()):
    if iteration == 1999:
        generator = load_model(
            "old_weights/wgan_CingulateSUVR_1999.h5"
        )
    else:
        generator = load_model(
            "old_weights/wgan_CingulateSUVR_29999.h5"
        )

    # generate control data
    if type == 0:
        synth = gan.test_generator(generator, type, 16)
    # generate subject data
    else:
        synth = gan.test_generator(generator, type, 11)

    # place generated data in a dataframe
    new_frame = pd.DataFrame(data=synth[0], columns=old_frame.columns)

    # concatenate new generated data to old frame passed into function
    frame_list = [old_frame, new_frame]
    final_frame = pd.concat(frame_list)

    return final_frame


# add sample now adds one more set of 11 aud samples or 16 control samples
# lets start at 30000 iterations of training

whole_suvr = pd.DataFrame(columns=aud_normal_x.columns)
whole_control = pd.DataFrame(columns=aud_normal_x.columns)

#1000 * 11 = 11000 synthetic aud samples
for i in range(0, 1000):
    whole_suvr = add_samples(whole_suvr, 1, 29999)

#1000 * 16 = 16000 synthetic control samples
for i in range(0, 1000):
    whole_control = add_samples(whole_control, 0, 29999)

whole_suvr = normalize_frame(whole_suvr)
whole_control = normalize_frame(whole_control)

whole_suvr.to_pickle("./whole_suvr_frame_29999.pkl")
whole_control.to_pickle("./whole_control_frame_29999.pkl")
