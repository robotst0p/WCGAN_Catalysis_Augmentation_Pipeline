import pandas as pd
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from scipy.special import rel_entr
from scipy.stats import entropy
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt


#maybe compare divergence values between successful candidates of 1999 and 30000 iterations
# original data no synthetic samples
aud_normal_x = pd.read_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/aud_frame_normal.pkl")
control_normal_x = pd.read_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/control_frame_normal.pkl")

success_cands = pd.read_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/randomforest_cand_x.pkl")
success_cands_y = pd.read_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/randomforest_cand_y.pkl")

print(success_cands_y)

pq_list = []
qp_list = []

# synthetic aud: synth_X_normal
# synthetic control: synth_control_normal
# original aud: aud_normal_x
# original control: control_normal_x


def kl_divergence(p, q):
    kl_pq = entropy(p, q)
    kl_qp = entropy(q, p)

    return kl_pq


def get_curve_data(axis, p_q):
    x, p = axis.get_lines()[p_q[0]].get_data()
    x, q = axis.get_lines()[p_q[1]].get_data()

    kl_pq = kl_divergence(p, q)

    return kl_pq


axis_list = []

divergence_frame = pd.DataFrame(
    columns=[
        "synthaud_originalaud",
        "synthaud_originalcontrol",
    ]
)
# l_suvr = 0
# k_suvr = 11

# l_control = 0
# k_control = 16
# for i in range(0, 100):
#     plot_frame_control = pd.DataFrame(columns=aud_normal_x.columns)
#     plot_frame_suvr = pd.DataFrame(columns=aud_normal_x.columns)

#     # select set of 16 samples from whole control frame
#     for column in aud_normal_x.columns:
#         plot_frame_control[column] = whole_control[column].iloc[l_control:k_control]

#     # select set of 11 samples from whole suvr frame
#     for column in aud_normal_x.columns:
#         plot_frame_suvr[column] = whole_suvr[column].iloc[l_suvr:k_suvr]

#     axis_list = []

plt.clf()
f, ([ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8]) = plt.subplots(nrows=2, ncols=4)
plt.title("HDAC/SUVR regional (Cingulate) value distribution (density plot)")

axis_list.append(ax1)
axis_list.append(ax2)
axis_list.append(ax3)
axis_list.append(ax4)
axis_list.append(ax5)
axis_list.append(ax6)
axis_list.append(ax7)
axis_list.append(ax8)

for i in range(0, len(axis_list)):
    sb.kdeplot(
        data=success_cands.iloc[:, i].tolist(),
        ax=axis_list[i],
        label="synthetic AUD",
        color="orange",
    )

for i in range(0, len(axis_list)):
    sb.kdeplot(
        data=aud_normal_x.iloc[:, i].tolist(),
        ax=axis_list[i],
        label="aud_original",
        color="red",
    )

for i in range(0, len(axis_list)):
    sb.kdeplot(
        data=control_normal_x.iloc[:, i].tolist(),
        ax=axis_list[i],
        label="control_original",
        color="blue",
    )

for axis in range(0, len(axis_list)):
    axis_list[axis].set(xlabel=aud_normal_x.columns[axis])

plt.draw()
plt.legend()
plt.show()

# synth aud: curve 0
# aud original: curve 1
# control original: curve 2
curvelist = [(0, 1), (0, 2)]

divergence_concat_frame = pd.DataFrame(columns=divergence_frame.columns)
for k in range(0, len(axis_list)):
    for i in range(0, len(divergence_frame.columns)):
        divergence_concat_frame.loc[
            k, divergence_frame.columns[i]
        ] = get_curve_data(axis_list[k], curvelist[i])

concat_list = [divergence_frame, divergence_concat_frame]

divergence_frame = pd.concat(concat_list)
divergence_frame.reset_index()
print(divergence_frame)

plt.clf()
plt.close()

divergence_frame.to_pickle("./succesful_divergence_frame_randomforest.pkl")
divergence_frame.to_csv("./succesful_divergence_frame_randomforest.csv")