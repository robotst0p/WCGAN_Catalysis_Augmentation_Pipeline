import pandas as pd
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import markers

# violin plot

# load dataframe of kl divergence values
divergence_frame_1999 = pd.read_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/divergence_frame_1999.pkl")
divergence_frame_29999 = pd.read_pickle("C:/Users/meyer/Desktop/SUVr_Analysis/saved_data/divergence_frame_29999.pkl")

print(divergence_frame_1999)
print(divergence_frame_29999)

# adjust index names to names of brain regions

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

# begin constructing frame for plotting data and converting divergence frame data from wide to long format for scatterplot
plot_frame_1999 = pd.DataFrame(columns=["KL Type", "Region", "Divergence"])
plot_frame_29999 = pd.DataFrame(columns=["KL Type", "Region", "Divergence"])

# establish different types of kl divergence between datatypes in scatterplotting frame
def kl_type(divergence_frame, plot_frame):
    l = 0
    k = 7

    for column in divergence_frame.columns:
        for i in range(l, k + 1):
            plot_frame.loc[i, "KL Type"] = column

        l += 8
        k += 8

    return plot_frame


plot_frame_1999 = kl_type(divergence_frame_1999, plot_frame_1999)
plot_frame_29999 = kl_type(divergence_frame_29999, plot_frame_29999)

# fill in brain region for each KL divergence value point
def region_type(divergence_frame, plot_frame):
    for i in range(0, len(plot_frame.index)):
        if i < 8:
            plot_frame.loc[i, "Region"] = region_list[i]
        else:
            plot_frame.loc[i, "Region"] = region_list[i % 8]

    return plot_frame


plot_frame_1999 = region_type(divergence_frame_1999, plot_frame_1999)
plot_frame_29999 = region_type(divergence_frame_29999, plot_frame_29999)

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


plot_frame_1999 = divergence_values(divergence_frame_1999, plot_frame_1999)
plot_frame_29999 = divergence_values(divergence_frame_29999, plot_frame_29999)

# print(plot_frame_1999)
# print(plot_frame_29999)

# plot kl divergence values
sns.set(font_scale=0.9)
sns.set_style(rc={"axes.facecolor": sns.color_palette("pastel")[8]})
sns.scatterplot(
    data=plot_frame_29999, x="Region", y="Divergence", hue="KL Type", alpha=1
).set(title="KL Divergence Values Across Cingulate Regions 29999 Iterations")

plt.show()

# clear plot for individual 8 region graphs
plt.clf()

# need different data frame for each region
# take the matching indexes from each frame and combine the rows into one frame: this will give us one frame for each of the 8 brain regions


def create_region_frame(
    plot_frame1,
    plot_frame2,
    region,
    new_frame=pd.DataFrame(),
    new_frame2=pd.DataFrame(),
):
    new_frame = plot_frame1.loc[plot_frame1["Region"] == region]
    new_frame2 = plot_frame2.loc[plot_frame2["Region"] == region]

    frame_list = [new_frame, new_frame2]
    final_frame = pd.concat(frame_list)

    final_frame = final_frame.reset_index()

    for i in range(0, 4):
        final_frame.loc[i, "Iteration"] = 1999

    for i in range(4, 8):
        final_frame.loc[i, "Iteration"] = 29999

    return final_frame


# region_list = ['ctx_lh_caudalanteriorcingulate','ctx_lh_isthmuscingulate','ctx_lh_posteriorcingulate','ctx_lh_rostralanteriorcingulate','ctx_rh_caudalanteriorcingulate','ctx_rh_isthmuscingulate','ctx_rh_posteriorcingulate','ctx_rh_rostralanteriorcingulate']
# print(plot_frame_1999)
# print(plot_frame_29999)

lh_caudalanterior = create_region_frame(
    plot_frame_1999, plot_frame_29999, "ctx_lh_caudalanteriorcingulate"
)
lh_isthmus = create_region_frame(
    plot_frame_1999, plot_frame_29999, "ctx_lh_isthmuscingulate"
)
lh_posterior = create_region_frame(
    plot_frame_1999, plot_frame_29999, "ctx_lh_posteriorcingulate"
)
lh_rostralanterior = create_region_frame(
    plot_frame_1999, plot_frame_29999, "ctx_lh_rostralanteriorcingulate"
)
rh_caudalanterior = create_region_frame(
    plot_frame_1999, plot_frame_29999, "ctx_rh_caudalanteriorcingulate"
)
rh_isthmus = create_region_frame(
    plot_frame_1999, plot_frame_29999, "ctx_rh_isthmuscingulate"
)
rh_posterior = create_region_frame(
    plot_frame_1999, plot_frame_29999, "ctx_rh_posteriorcingulate"
)
rh_rostralanterior = create_region_frame(
    plot_frame_1999, plot_frame_29999, "ctx_rh_rostralanteriorcingulate"
)

# create scatter plots for kl divergences values split by regions and specified by type and generator iteration number
f, ([ax1, ax2, ax3, ax4], [ax5, ax6, ax7, ax8]) = plt.subplots(nrows=2, ncols=4)
plt.title("KL Divergence Values Across Cingulate Regions by Generator Iterations")
# plot kl divergence values
sns.set(font_scale=0.8)
sns.set_style(rc={"axes.facecolor": sns.color_palette("pastel")[8]})
sns.scatterplot(
    data=lh_caudalanterior,
    x="KL Type",
    y="Divergence",
    hue="Iteration",
    alpha=1,
    palette="Set2",
    style="KL Type",
    ax=ax1,
).set(title="LH Caudalanterior Cingulate", xticklabels=[], xlabel=None)
ax1.legend(bbox_to_anchor=(-0.74, 1), loc="upper left", borderaxespad=0, fontsize=7.6)

sns.scatterplot(
    data=lh_isthmus,
    x="KL Type",
    y="Divergence",
    hue="Iteration",
    alpha=1,
    palette="Set2",
    style="KL Type",
    ax=ax2,
    legend=False,
).set(title="LH Isthmus Cingulate", xticklabels=[], xlabel=None)
sns.scatterplot(
    data=lh_posterior,
    x="KL Type",
    y="Divergence",
    hue="Iteration",
    alpha=1,
    palette="Set2",
    style="KL Type",
    ax=ax3,
    legend=False,
).set(title="LH Posterior Cingulate", xticklabels=[], xlabel=None)
sns.scatterplot(
    data=lh_rostralanterior,
    x="KL Type",
    y="Divergence",
    hue="Iteration",
    alpha=1,
    palette="Set2",
    style="KL Type",
    ax=ax4,
    legend=False,
).set(title="LH Rostralanterior Cingulate", xticklabels=[], xlabel=None)
sns.scatterplot(
    data=rh_caudalanterior,
    x="KL Type",
    y="Divergence",
    hue="Iteration",
    alpha=1,
    palette="Set2",
    style="KL Type",
    ax=ax5,
    legend=False,
).set(title="RH Caudalanterior Cingulate", xticklabels=[], xlabel=None)
sns.scatterplot(
    data=rh_isthmus,
    x="KL Type",
    y="Divergence",
    hue="Iteration",
    alpha=1,
    palette="Set2",
    style="KL Type",
    ax=ax6,
    legend=False,
).set(title="RH Isthmus Cingulate", xticklabels=[], xlabel=None)
sns.scatterplot(
    data=rh_posterior,
    x="KL Type",
    y="Divergence",
    hue="Iteration",
    alpha=1,
    palette="Set2",
    style="KL Type",
    ax=ax7,
    legend=False,
).set(title="RH Posterior Cingulate", xticklabels=[], xlabel=None)
sns.scatterplot(
    data=rh_rostralanterior,
    x="KL Type",
    y="Divergence",
    hue="Iteration",
    alpha=1,
    palette="Set2",
    style="KL Type",
    ax=ax8,
    legend=False,
).set(title="RH Rostralanterior Cingulate", xticklabels=[], xlabel=None)

plt.show()

# clear plot for individual 8 region graphs
plt.clf()
