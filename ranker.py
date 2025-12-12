import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
import numpy as np
import logging
import itertools
import json
import sys

logger = logging.getLogger("cmdstanpy")
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

use_fast = "--fast" in sys.argv[1:]

for gamemode in [
    "catch",
    "taiko",
    "mania",
    "osu",
]:

    user_id_to_username = {}
    with open("./processed_score_files/{}_users.csv".format(gamemode)) as f:
        for line in f:
            user_id, username = line.split(",", maxsplit=1)
            user_id = int(user_id)
            username = username[1:-2]
            user_id_to_username[user_id] = username

    user_mapping = {}
    user_mapping_reverse = {}
    map_mapping = {}
    map_mapping_reverse = {}

    scores = []

    model = CmdStanModel(
        stan_file="./models/ranker.stan", cpp_options={"STAN_THREADS": True}
    )

    with open("./processed_score_files/{}.json".format(gamemode)) as f:
        for user_id, map_id, mods, score in json.load(f):
            mapped_map = map_mapping.get(str((map_id, mods)))
            if mapped_map is None:
                id = len(map_mapping) + 1
                map_mapping[str((map_id, mods))] = id
                map_mapping_reverse[id] = str((map_id, mods))
                mapped_map = map_mapping.get(str((map_id, mods)))

            mapped_user = user_mapping.get(user_id)
            if mapped_user is None:
                id = len(user_mapping) + 1
                user_mapping[user_id] = id
                user_mapping_reverse[id] = user_id
                mapped_user = user_mapping.get(user_id)

            scores.append((mapped_map, mapped_user, score))

    scores.sort(key=lambda x: (x[0], x[2]))

    score_ties = []
    for _, items in itertools.groupby(enumerate(scores, 1), lambda x: x[1][0]):
        items = list(items)

        for (_, (map_x, _, score_x)), (_, (map_y, _, score_y)) in itertools.pairwise(
            items
        ):
            score_ties.append(1 if (map_x, score_x) == (map_y, score_y) else 0)
        score_ties.append(0)

    scores_per_map = [0] * len(map_mapping)
    for m, p, _ in scores:
        scores_per_map[m - 1] += 1

    data = {
        "M": len(map_mapping),
        "S": len(scores),
        "P": len(user_mapping),
        "scores_per_map": scores_per_map,
        "score_to_player": [p for (_, p, _) in scores],
        "score_ties": score_ties,
    }

    if use_fast:
        fit = model.optimize(
            data=data,
            show_console=True,
            refresh=1,
            require_converged=False,
            iter=100_000,
            inits={
                "beta": np.full(data["P"], 0.001),
            },
        )
        fit_beta_mean = - np.log(fit.beta)
        fit_beta_mean = 2 + (fit_beta_mean-np.mean(fit_beta_mean))/np.var(fit_beta_mean)
        fit_beta_rankings = (
            len(fit_beta_mean)
            - 1
            - np.argsort(np.argsort(fit.beta, axis=-1), axis=-1)
        ) / (len(fit_beta_mean) - 1)
    else:
        fit = model.sample(
            data=data,
            show_progress=True,
            refresh=1,
            chains=1,
            inits={
                "beta": np.full(data["P"], 0.001),
            },
            threads_per_chain=8,
        )
        fit_beta_mean = - np.log(np.mean(fit.beta, axis=0))
        fit_beta_mean = 2 + (fit_beta_mean-np.mean(fit_beta_mean))/np.var(fit_beta_mean)
        fit_beta_rankings = (
            len(fit_beta_mean)
            - 1
            - np.mean(np.argsort(np.argsort(fit.beta, axis=-1), axis=-1), axis=0)
        ) / (len(fit_beta_mean) - 1)

    print("sampling done")

    user_skills = []

    for k, (skill, rank) in enumerate(zip(fit_beta_mean, fit_beta_rankings), 1):
        user_skills.append((user_mapping_reverse[k], skill, rank))

    user_skills.sort(key=lambda x: np.mean(x[2]), reverse=True)

    with open("{}_ranking.txt".format(gamemode), "w") as f:
        for user, skill, rank in user_skills:
            print(user, skill, rank, user_id_to_username[user], sep=",", file=f)
    print("saving done")
