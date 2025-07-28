import pandas as pd
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import markers
from matplotlib import font_manager

# load in whole divergence frame of 800 rows
# frame is 800 rows by 4 columns
divergence_frame = pd.read_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/whole_divergence_frame_29999.pkl")

print(divergence_frame)

# adjust dataframe for region column with list of regions
region_list = [
    "ctx_lh_caudalanteriorcingulate",
    "ctx_lh_isthmuscingulate",
    "ctx_lh_posteriorcingulate",
    "ctx_lh_rostralanteriorcingulate",
    "ctx_rh_caudalanteriorcingulate",
    "ctx_rh_isthmuscingulate",
    "ctx_rh_posteriorcingulate",
    "ctx_rh_rostralanteriorcingulate",
]

# begin constructing frame for plotting data and converting divergence frame data from wide to long format for violin plot
plot_frame = pd.DataFrame(columns=["KL Type", "Region", "Divergence"])

def kl_type(divergence_frame, plot_frame):
    l = 0
    k = 8000

    for column in divergence_frame.columns:
        for i in range(l, k):
            plot_frame.loc[i, "KL Type"] = column

        l += 8000
        k += 8000
    return plot_frame



plot_frame = kl_type(divergence_frame, plot_frame)

# fill in brain region for each KL divergence value point
def region_type(divergence_frame, plot_frame):
    for i in range(0, len(plot_frame.index)):
        if i < 8:
            plot_frame.loc[i, "Region"] = region_list[i]
        else:
            plot_frame.loc[i, "Region"] = region_list[i % 8]

    return plot_frame


plot_frame = region_type(divergence_frame, plot_frame)

# fill in the kl divergence values from the original divergence frame for each kl type and brain region
def divergence_values(divergence_frame, plot_frame):
    kl_point_list = []
    for column in divergence_frame.columns:
        current_list = divergence_frame.loc[:, column].tolist()
        for i in range(0, len(current_list)):
            kl_point_list.append(current_list[i])

    for i in range(0, len(plot_frame.index)):
        plot_frame.loc[i, "Divergence"] = kl_point_list[i]

    return plot_frame


plot_frame = divergence_values(divergence_frame, plot_frame)

print(plot_frame)

# need different data frame for each region
# take the matching indexes from each frame and combine the rows into one frame: this will give us one frame for each of the 8 brain regions
def rename_divergence_types(frame):
    frame['KL Type'].replace('synthaud_originalaud', 'Synth AUD to OG AUD', inplace=True)
    frame['KL Type'].replace('synthcontrol_originalcontrol', 'Synth Control to OG Control', inplace=True)
    frame['KL Type'].replace('synthaud_originalcontrol', 'Synth AUD to OG Control', inplace=True)
    frame['KL Type'].replace('synthcontrol_originalaud', 'Synth Control to OG AUD', inplace=True)

    return (frame)
    
def create_region_frame(
    plot_frame1,
    region,
    new_frame=pd.DataFrame(),
    new_frame2=pd.DataFrame(),
):
    new_frame = plot_frame1.loc[plot_frame1["Region"] == region]

    final_frame = new_frame.reset_index()

    final_frame = rename_divergence_types(final_frame)

    return final_frame

lh_caudalanterior = create_region_frame(plot_frame, "ctx_lh_caudalanteriorcingulate")
lh_isthmus = create_region_frame(plot_frame, "ctx_lh_isthmuscingulate")
lh_posterior = create_region_frame(plot_frame, "ctx_lh_posteriorcingulate")
lh_rostralanterior = create_region_frame(plot_frame, "ctx_lh_rostralanteriorcingulate")
rh_caudalanterior = create_region_frame(plot_frame, "ctx_rh_caudalanteriorcingulate")
rh_isthmus = create_region_frame(plot_frame, "ctx_rh_isthmuscingulate")
rh_posterior = create_region_frame(plot_frame, "ctx_rh_posteriorcingulate")
rh_rostralanterior = create_region_frame(plot_frame, "ctx_rh_rostralanteriorcingulate")

brain_frames = [
    lh_caudalanterior,
    lh_isthmus,
    lh_posterior,
    lh_rostralanterior,
    rh_caudalanterior,
    rh_isthmus,
    rh_posterior,
    rh_rostralanterior,
]

# need to convert frame column datatypes to float
def to_float(plot_frame):
    plot_frame["Divergence"] = plot_frame["Divergence"].astype(float)

    return plot_frame


for frame in brain_frames:
    frame = to_float(frame)


# create violin plots for divergence values by region, grouped by kl type
f, ([ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8]) = plt.subplots(nrows=2, ncols=4)
plt.title("KL Divergence Values Across Cingulate Regions")
# plot kl divergence values
sns.set(font_scale=1.5)
# sns.set_style(rc={"axes.facecolor": sns.color_palette("pastel")[8]})

#Synth AUD to OG AUD
#Synth AUD to OG Control
#Synth Control to OG Control
#Synth Control to OG AUD

sns.violinplot(
    data=lh_caudalanterior,
    x="KL Type",
    y="Divergence",
    hue="KL Type",
    palette={
        "Synth AUD to OG AUD": "orange",
        "Synth AUD to OG Control": "red",
        "Synth Control to OG Control": "blue",
        "Synth Control to OG AUD": "cyan"
    },
    order=[
        "Synth AUD to OG AUD",
        "Synth AUD to OG Control",
        " ",
        "Synth Control to OG Control",
        "Synth Control to OG AUD"
    ],
    legend = True,
    ax=ax1,
).set(xticklabels=[], xlabel=None)
ax1.legend(bbox_to_anchor=(-0.72, 1.32), loc="upper left", borderaxespad=0, fontsize=29, ncol = 4, frameon = False)
sns.violinplot(
    data=lh_isthmus,
    x="KL Type",
    y="Divergence",
    hue="KL Type",
    palette={
        "Synth AUD to OG AUD": "orange",
        "Synth AUD to OG Control": "red",
        "Synth Control to OG Control": "blue",
        "Synth Control to OG AUD": "cyan"
    },
    order=[
        "Synth AUD to OG AUD",
        "Synth AUD to OG Control",
        " ",
        "Synth Control to OG Control",
        "Synth Control to OG AUD"
    ],
    ax=ax2,
    legend=False,
).set(xticklabels=[], xlabel=None)
ax2.set_ylabel(None)
sns.violinplot(
    data=lh_posterior,
    x="KL Type",
    y="Divergence",
    hue="KL Type",
    palette={
        "Synth AUD to OG AUD": "orange",
        "Synth AUD to OG Control": "red",
        "Synth Control to OG Control": "blue",
        "Synth Control to OG AUD": "cyan"
    },
    order=[
        "Synth AUD to OG AUD",
        "Synth AUD to OG Control",
        " ",
        "Synth Control to OG Control",
        "Synth Control to OG AUD"
    ],
    ax=ax3,
    legend=False,
).set(xticklabels=[], xlabel=None)
ax3.set_ylabel(None)
sns.violinplot(
    data=lh_rostralanterior,
    x="KL Type",
    y="Divergence",
    hue="KL Type",
    palette={
        "Synth AUD to OG AUD": "orange",
        "Synth AUD to OG Control": "red",
        "Synth Control to OG Control": "blue",
        "Synth Control to OG AUD": "cyan"
    },
    order=[
        "Synth AUD to OG AUD",
        "Synth AUD to OG Control",
        " ",
        "Synth Control to OG Control",
        "Synth Control to OG AUD"
    ],
    ax=ax4,
    legend=False,
).set(xticklabels=[], xlabel=None)
ax4.set_ylabel(None)
sns.violinplot(
    data=rh_caudalanterior,
    x="KL Type",
    y="Divergence",
    hue="KL Type",
    palette={
        "Synth AUD to OG AUD": "orange",
        "Synth AUD to OG Control": "red",
        "Synth Control to OG Control": "blue",
        "Synth Control to OG AUD": "cyan"
    },
    order=[
        "Synth AUD to OG AUD",
        "Synth AUD to OG Control",
        " ",
        "Synth Control to OG Control",
        "Synth Control to OG AUD"
    ],
    ax=ax5,
    legend=False,
).set(xticklabels=[], xlabel=None)
sns.violinplot(
    data=rh_isthmus,
    x="KL Type",
    y="Divergence",
    hue="KL Type",
    palette={
        "Synth AUD to OG AUD": "orange",
        "Synth AUD to OG Control": "red",
        "Synth Control to OG Control": "blue",
        "Synth Control to OG AUD": "cyan"
    },
    order=[
        "Synth AUD to OG AUD",
        "Synth AUD to OG Control",
        " ",
        "Synth Control to OG Control",
        "Synth Control to OG AUD"
    ],
    ax=ax6,
    legend=False,
).set(xticklabels=[], xlabel=None)
ax6.set_ylabel(None)
sns.violinplot(
    data=rh_posterior,
    x="KL Type",
    y="Divergence",
    hue="KL Type",
    palette={
        "Synth AUD to OG AUD": "orange",
        "Synth AUD to OG Control": "red",
        "Synth Control to OG Control": "blue",
        "Synth Control to OG AUD": "cyan"
    },
    order=[
        "Synth AUD to OG AUD",
        "Synth AUD to OG Control",
        " ",
        "Synth Control to OG Control",
        "Synth Control to OG AUD"
    ],
    ax=ax7,
    legend=False,
).set(xticklabels=[], xlabel=None)
ax7.set_ylabel(None)
sns.violinplot(
    data=rh_rostralanterior,
    x="KL Type",
    y="Divergence",
    hue="KL Type",
    width = 1,
    palette={
        "Synth AUD to OG AUD": "orange",
        "Synth AUD to OG Control": "red",
        "Synth Control to OG Control": "blue",
        "Synth Control to OG AUD": "cyan"
    },
    order=[
        "Synth AUD to OG AUD",
        "Synth AUD to OG Control",
        " ",
        "Synth Control to OG Control",
        "Synth Control to OG AUD"
    ],
    ax=ax8,
    legend=False,
).set(xticklabels=[], xlabel=None)

ax1.set_title("LH Caudalanterior Cingulate", fontsize=30)
ax2.set_title("LH Isthmus Cingulate", fontsize=30)
ax3.set_title("LH Posterior Cingulate", fontsize=30)
ax4.set_title("LH Rostralanterior Cingulate", fontsize=30)
ax5.set_title("RH Caudalanterior Cingulate", fontsize=30)
ax6.set_title("RH Isthmus Cingulate", fontsize=30)
ax7.set_title("RH Posterior Cingulate", fontsize=30)
ax8.set_title("RH Rostralanterior Cingulate", fontsize=30)

ax8.set_ylabel(None)
ax1.set_ylabel("KL Divergence", fontsize = 35)
ax5.set_ylabel("KL Divergence", fontsize = 35)

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

for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
    ax.tick_params(axis='y', labelsize=29)

plt.show()

# clear plot for individual 8 region graphs
plt.clf()
