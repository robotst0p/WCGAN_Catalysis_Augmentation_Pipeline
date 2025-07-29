import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from lib import gan_architecture as gan
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from scipy.special import rel_entr
from scipy.stats import entropy
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

def normalize_frame(frame):
    scaler2 = StandardScaler()

    synth_model = scaler2.fit(frame)
    synth_frame_normal = pd.DataFrame(
        synth_model.transform(frame), columns=frame.columns
    )

    return synth_frame_normal

average_synth_1999 = pd.read_pickle("../saved_data/average_synth_1999.pkl")
average_synth_29999 = pd.read_pickle("../saved_data/average_synth_29999.pkl")

average_control_1999 = pd.read_pickle("../saved_data/average_control_1999.pkl")
average_control_29999 = pd.read_pickle("../saved_data/average_control_29999.pkl")

aud_normal_x = pd.read_pickle("../saved_data/aud_frame_normal.pkl")
control_normal_x = pd.read_pickle("../saved_data/control_frame_normal.pkl")

svm_cands = pd.read_pickle("../saved_data/final figures data/Figure 6 (Successful Candidates KDE Plot)/new_svm_cand_x.pkl")
xg_cands = pd.read_pickle("../saved_data/final figures data/Figure 6 (Successful Candidates KDE Plot)/new_xgb_cand_x.pkl")
rf_cands = pd.read_pickle("../saved_data/new_rf_cand_x.pkl")

generator_1999 = load_model(
    "old_weights/wgan_CingulateSUVR_1999.h5"
)

generator_29999 = load_model(
    "old_weights/wgan_CingulateSUVR_29999.h5"
)

synth_aud1999 = gan.test_generator(generator_1999, 1, 11)

synth_control1999 = gan.test_generator(generator_1999, 0, 16)

synth_aud29999 = gan.test_generator(generator_29999, 1, 11)

synth_control29999 = gan.test_generator(generator_29999, 0, 16)

synthaudframe_1999 = pd.DataFrame(
    data=synth_aud1999[0], columns=aud_normal_x.columns
)
synthcontrolframe_1999 = pd.DataFrame(
    data=synth_control1999[0], columns=aud_normal_x.columns
)

synthaudframe_29999 = pd.DataFrame(
    data=synth_aud29999[0], columns=aud_normal_x.columns
)

synthcontrolframe_29999 = pd.DataFrame(
    data=synth_control29999[0], columns=aud_normal_x.columns
)

#normalize generated data
synthaudframe_1999 = normalize_frame(synthaudframe_1999)
synthcontrolframe_1999 = normalize_frame(synthcontrolframe_1999)

synthaudframe_29999 = normalize_frame(synthaudframe_29999)
synthcontrolframe_29999 = normalize_frame(synthcontrolframe_29999)

axis_list = []
plt.clf()
f, ([ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8]) = plt.subplots(nrows=2, ncols=4)

axis_list.append(ax1)
axis_list.append(ax2)
axis_list.append(ax3)
axis_list.append(ax4)
axis_list.append(ax5)
axis_list.append(ax6)
axis_list.append(ax7)
axis_list.append(ax8)

# #uncomment if regular kde plot and not sucessful cands
# for i in range(0, len(axis_list)):
#     sb.kdeplot(
#         data=average_control_29999.iloc[:, i].tolist(),
#         ax=axis_list[i],
#         label="synthetic CONTROL",
#         color="cyan",
#     )
    
for i in range(0, len(axis_list)):
    sb.kdeplot(
        data=average_synth_1999.iloc[:, i].tolist(),
        ax=axis_list[i],
        label="Synthetic AUD",
        color="orange",
    )

for i in range(0, len(axis_list)):
    sb.kdeplot(
        data=aud_normal_x.iloc[:, i].tolist(),
        ax=axis_list[i],
        label="Original AUD",
        color="red",
    )

for i in range(0, len(axis_list)):
    sb.kdeplot(
        data=control_normal_x.iloc[:, i].tolist(),
        ax=axis_list[i],
        label="Original Control",
        color="blue",
    )

ax1.set_xlabel("lh-caudalanterior", fontsize = 15)
ax2.set_xlabel("lh-isthmus", fontsize = 15)
ax3.set_xlabel("lh-posterior", fontsize = 15)
ax4.set_xlabel("lh-rostralanterior", fontsize = 15)
ax5.set_xlabel("rh-caudalanterior", fontsize = 15)
ax6.set_xlabel("rh-isthmus", fontsize = 15)
ax7.set_xlabel("rh-posterior", fontsize = 15)
ax8.set_xlabel("rh-rostralanterior", fontsize = 15)

ax2.set_ylabel(None)
ax3.set_ylabel(None)
ax4.set_ylabel(None)
ax6.set_ylabel(None)
ax7.set_ylabel(None)
ax8.set_ylabel(None)

ax1.set_ylabel("Density", fontsize = 40)
ax5.set_ylabel("Density", fontsize = 40)

ax1.tick_params(axis='both', which='major', labelsize=25)
ax1.tick_params(axis='both', which='minor', labelsize=25)

ax2.tick_params(axis='both', which='major', labelsize=25)
ax2.tick_params(axis='both', which='minor', labelsize=25)

ax3.tick_params(axis='both', which='major', labelsize=25)
ax3.tick_params(axis='both', which='minor', labelsize=25)

ax4.tick_params(axis='both', which='major', labelsize=25)
ax4.tick_params(axis='both', which='minor', labelsize=25)

ax5.tick_params(axis='both', which='major', labelsize=25)
ax5.tick_params(axis='both', which='minor', labelsize=25)

ax6.tick_params(axis='both', which='major', labelsize=25)
ax6.tick_params(axis='both', which='minor', labelsize=25)

ax7.tick_params(axis='both', which='major', labelsize=25)
ax7.tick_params(axis='both', which='minor', labelsize=25)

ax8.tick_params(axis='both', which='major', labelsize=25)
ax8.tick_params(axis='both', which='minor', labelsize=25)

ax1.legend()
sb.move_legend(
    ax1, "lower center",
    bbox_to_anchor=(2.2, 1), ncol=4, title=None, frameon=False, fontsize = 35,
)

plt.draw()
#plt.legend()
plt.show()
