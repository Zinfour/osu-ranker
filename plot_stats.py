import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["font.family"] = "Source Serif 4"
plt.rcParams["font.size"] = 20


img = imread('./my_cmap.png')
colors_from_img = img[0, :, :]
my_cmap = LinearSegmentedColormap.from_list('my_cmap', colors_from_img, N=colors_from_img.shape[0])

for gamemode in [
    "catch",
    "taiko",
    "mania",
    "osu",
]:
    
    fig = plt.figure(figsize=(8, 8), dpi=300)
    fig.set_facecolor("#161616")
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor("#161616")
    user_id_to_pp = {}
    user_id_to_skill = {}
    with open(f"./outputs/{gamemode}_ranking.csv") as f:
        for line in f:
            user_id, skill, average_ranking, username, scores, normal_total_pp = line.strip().split(",", maxsplit=5)
            if int(scores) == 0:
                continue
            user_id_to_skill[int(user_id)] = float(skill)
            user_id_to_pp[int(user_id)] = float(normal_total_pp)

    a1 = list(user_id_to_pp.keys())
    a1.sort(key=lambda user_id: user_id_to_pp[user_id])
    pp_pos = {}
    for i, user_id in enumerate(a1):
        pp_pos[user_id] = i

    a2 = list(user_id_to_skill.keys())
    a2.sort(key=lambda user_id: user_id_to_skill[user_id])
    skill_pos = {}
    for i, user_id in enumerate(a2):
        skill_pos[user_id] = i

    x = []
    y = []
    for user_id, pp in user_id_to_pp.items():
        x.append(user_id_to_pp[user_id])
        y.append(user_id_to_skill[user_id])

    ax.margins(x=0.01, y=0.01)
    ax.hexbin(
        x,
        y,
        bins="log",
        mincnt=1,
        gridsize=(int(np.sqrt(3) * 50), 50),
        cmap=my_cmap,
        linewidths=0.0,
    )
    ax.set_title(f"{gamemode}", fontsize=30)
    ax.title
    ax.set_xlabel("pp")
    ax.set_ylabel("skill")

    ax.spines["bottom"].set_color("#f1f1f1")
    ax.spines["top"].set_color("#f1f1f1")
    ax.spines["right"].set_color("#f1f1f1")
    ax.spines["left"].set_color("#f1f1f1")
    ax.tick_params(axis="x", colors="#f1f1f1")
    ax.tick_params(axis="y", colors="#f1f1f1")
    ax.yaxis.label.set_color("#f1f1f1")
    ax.xaxis.label.set_color("#f1f1f1")
    ax.title.set_color("#f1f1f1")
    fig.tight_layout(pad=1)
    fig.subplots_adjust(left=0.1, right=0.9)
    fig.savefig(f"./outputs/{gamemode}_plot.svg", dpi=300)
