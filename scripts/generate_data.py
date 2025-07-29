import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from lib import gan_architecture as gan
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

#script generates synthetic samples using trained models and averages them to match the original sample size

# function to normalize a passed in frame
def normalize_frame(frame):
    scaler2 = StandardScaler()

    synth_model = scaler2.fit(frame)
    synth_frame_normal = pd.DataFrame(
        synth_model.transform(frame), columns=frame.columns
    )

    return synth_frame_normal


# load original dataframe as reference
aud_normal_x = pd.read_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/aud_frame_normal.pkl")

# generator trained for 1999 iterations
generator_1999 = load_model(
    "C:/Users/meyer/Desktop/SUVr_Analysis/scripts/old_weights/wgan_CingulateSUVR_1999.h5"
)

# generator trained for 199999 iterations
generator_29999 = load_model(
    "C:/Users/meyer/Desktop/SUVr_Analysis/scripts/old_weights/wgan_CingulateSUVR_29999.h5"
)

# generate subject data for 1999 iterations
synthetic_suvr_1999 = gan.test_generator(generator_1999, 1, 11)

# generate control data for 1999 iterations
synthetic_control_1999 = gan.test_generator(generator_1999, 0, 16)

# generate synthetic subject data for 199999 iterations matching original dataset sample size
synthetic_suvr_29999 = gan.test_generator(generator_29999, 1, 11)

# generate synthetic subject data for 199999 iterations matching original dataset sample size
synthetic_control_29999 = gan.test_generator(generator_29999, 0, 16)

# establish initial frames for control and subject generation
synth_frame_x_1999 = pd.DataFrame(
    data=synthetic_suvr_1999[0], columns=aud_normal_x.columns
)
synth_frame_x_29999 = pd.DataFrame(
    data=synthetic_suvr_29999[0], columns=aud_normal_x.columns
)

synth_control_x_1999 = pd.DataFrame(
    data=synthetic_control_1999[0], columns=aud_normal_x.columns
)
synth_control_x_29999 = pd.DataFrame(
    data=synthetic_control_29999[0], columns=aud_normal_x.columns
)

synth_frame_x_1999 = normalize_frame(synth_frame_x_1999)
synth_frame_x_29999 = normalize_frame(synth_frame_x_29999)

synth_control_x_1999 = normalize_frame(synth_control_x_1999)
synth_control_x_29999 = normalize_frame(synth_control_x_29999)

synth_frame_x_1999.to_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/gen_suvr_1999.pkl")
synth_frame_x_29999.to_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/gen_suvr_29999.pkl")

synth_control_x_1999.to_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/gen_control_1999.pkl")
synth_control_x_29999.to_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/gen_control_29999.pkl")

# generate 100 sets of synthetic samples, average them, add them to one frame until desired count is reached, then normalize and analyze
# normalize them at the end
def add_samples(old_frame, type, iteration, new_frame=pd.DataFrame()):
    if iteration == 1999:
        generator = load_model(
            "C:/Users/meyer/Desktop/SUVr_Analysis/scripts/old_weights/wgan_CingulateSUVR_1999.h5"
        )
    else:
        generator = load_model(
            "C:/Users/meyer/Desktop/SUVr_Analysis/scripts/old_weights/wgan_CingulateSUVR_29999.h5"
        )

    # generate control data
    synth = gan.test_generator(generator, type, 100)
    # generate subject data

    # place generated data in a dataframe
    new_frame = pd.DataFrame(data=synth[0], columns=old_frame.columns)
    concat_frame = pd.DataFrame(columns=synth_control_x_29999.columns)

    # calculate the mean of the 100 generated samples for each region and place it as a new sample in concat frame
    for column in new_frame.columns:
        concat_frame.loc[0, column] = new_frame.loc[:, column].mean()

    # concatenate new generated data to old frame passed into function
    frame_list = [old_frame, concat_frame]
    final_frame = pd.concat(frame_list)

    return final_frame


# empty dataframes to put the average samples into with the same columns as the original dataframes
average_synth_1999 = pd.DataFrame(columns=synth_frame_x_1999.columns)
average_synth_29999 = pd.DataFrame(columns=synth_frame_x_1999.columns)
average_control_1999 = pd.DataFrame(columns=synth_frame_x_1999.columns)
average_control_29999 = pd.DataFrame(columns=synth_frame_x_1999.columns)

# for synthetic subjects, generate 1100 and average every 100 for a total of 11 synthetic subjects
# average divergence not features
for i in range(0, 11):
    average_synth_1999 = add_samples(average_synth_1999, 1, 1999)
    average_synth_29999 = add_samples(average_synth_29999, 1, 29999)

# for synthetic controls, generate 1600 and average every 100 for a total of 16 synthetic subjects
for i in range(0, 16):
    average_control_1999 = add_samples(average_control_1999, 0, 1999)
    average_control_29999 = add_samples(average_control_29999, 0, 29999)

# fix indexes after concatenation
average_synth_1999.reset_index()
average_synth_29999.reset_index()
average_control_1999.reset_index()
average_control_29999.reset_index()

# normalize the data in the average frames
average_synth_1999 = normalize_frame(average_synth_1999)
average_synth_29999 = normalize_frame(average_synth_29999)
average_control_1999 = normalize_frame(average_control_1999)
average_control_29999 = normalize_frame(average_control_29999)

# pickle the frames for divergence and plotting analysis
average_synth_1999.to_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/average_synth_1999.pkl")
average_synth_29999.to_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/average_synth_29999.pkl")
average_control_1999.to_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/average_control_1999.pkl")
average_control_29999.to_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/average_control_29999.pkl")
