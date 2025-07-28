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


# original data no synthetic samples
aud_normal_x = pd.read_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/aud_frame_normal.pkl")
control_normal_x = pd.read_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/control_frame_normal.pkl")

synth_suvr_x_1999 = pd.read_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/gen_suvr_1999.pkl")
synth_suvr_x_29999 = pd.read_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/gen_suvr_29999.pkl")

synth_control_x_1999 = pd.read_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/gen_control_1999.pkl")
synth_control_x_29999 = pd.read_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/gen_control_29999.pkl")

average_control_1999 = pd.read_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/average_control_1999.pkl")
average_control_29999 = pd.read_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/average_control_29999.pkl")

average_synth_1999 = pd.read_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/average_synth_1999.pkl")
average_synth_29999 = pd.read_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/average_synth_29999.pkl")

#change the path to load 1999 iterations instead of 29999
whole_aud = pd.read_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/scripts/whole_suvr_frame_29999.pkl")
whole_control = pd.read_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/scripts/whole_control_frame_29999.pkl")


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
        "synthcontrol_originalcontrol",
        "synthaud_originalcontrol",
        "synthcontrol_originalaud",
    ]
)

l_suvr = 0
k_suvr = 11

l_control = 0
k_control = 16

for i in range(0, 10000):
    plot_frame_control = pd.DataFrame(columns=aud_normal_x.columns)
    plot_frame_aud = pd.DataFrame(columns=aud_normal_x.columns)

    # select set of 16 samples from whole control frame
    for column in aud_normal_x.columns:
        plot_frame_control[column] = whole_control[column].iloc[l_control:k_control]

    # select set of 11 samples from whole suvr frame
    for column in aud_normal_x.columns:
        plot_frame_aud[column] = whole_aud[column].iloc[l_suvr:k_suvr]

    axis_list = []
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
            data=plot_frame_control.iloc[:, i].tolist(),
            ax=axis_list[i],
            label="synthetic CONTROL",
            color="cyan",
        )

    for i in range(0, len(axis_list)):
        sb.kdeplot(
            data=plot_frame_aud.iloc[:, i].tolist(),
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
    # plt.show()

    # synth control: curve 0
    # synthetic aud: curve 1
    # aud original: curve 2
    # control original: curve 3

    curvelist = [(1, 2), (0, 3), (1, 3), (0, 2)]

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

    l_suvr += 11
    k_suvr += 11

    l_control += 16
    k_control += 16
    plt.clf()
    plt.close()

divergence_frame.to_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/whole_divergence_frame_29999.pkl")

# f, ([ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8]) = plt.subplots(nrows = 2, ncols = 4)
# plt.title("HDAC/SUVR regional (Cingulate) value distribution (density plot)")

# axis_list.append(ax1)
# axis_list.append(ax2)
# axis_list.append(ax3)
# axis_list.append(ax4)
# axis_list.append(ax5)
# axis_list.append(ax6)
# axis_list.append(ax7)
# axis_list.append(ax8)

# for i in range(0, len(axis_list)):
#     sb.kdeplot(data = synth_control_x_29999.iloc[:,i].tolist(), ax = axis_list[i], label = 'synthetic CONTROL', color = 'cyan')

# for i in range(0, len(axis_list)):
#     sb.kdeplot(data = synth_suvr_x_29999.iloc[:,i].tolist(), ax = axis_list[i], label = 'synthetic AUD', color = 'orange')

# for i in range(0, len(axis_list)):
#     sb.kdeplot(data = aud_normal_x.iloc[:,i].tolist(), ax = axis_list[i], label = 'aud_original', color = 'red')

# for i in range(0, len(axis_list)):
#     sb.kdeplot(data = control_normal_x.iloc[:,i].tolist(), ax = axis_list[i], label = 'control_original', color = 'blue')

# for axis in range(0, len(axis_list)):
#         axis_list[axis].set(xlabel = aud_normal_x.columns[axis])

# plt.draw()
# plt.legend()
# plt.show()

# divergence_frame = pd.DataFrame(columns = ['synthaud_originalaud', 'synthcontrol_originalcontrol', 'synthaud_originalcontrol','synthcontrol_originalaud'])
# #                                         [(1,2), (2,1), (0,3), (3,0), (1,3), (3,1), (1,0), (0,1), (0,2), (2,0)]
# #                                         [(1,2),(0,3),(1,3),(0,2)]
# #synth control: curve 0
# #synthetic aud: curve 1
# #aud original: curve 2
# #control original: curve 3

# curvelist = [(1,2), (0,3), (1,3), (0,2)]

# for k in range(0, len(axis_list)):
#     for i in range(0, len(divergence_frame.columns)):
#         divergence_frame.loc[k, divergence_frame.columns[i]] = get_curve_data(axis_list[k], curvelist[i])


# print(divergence_frame)
# plt.clf()

# divergence_frame.to_pickle("./divergence_frame_29999.pkl")

# divergence_frame.to_csv("./divergence_frame_29999.csv")
