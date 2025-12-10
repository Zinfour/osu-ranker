# osu! ranker
Ranks the top 10000 players of [osu](https://osu.ppy.sh/) based on leaderboard positions.
Run `cargo run --release` to download and process the scores and then `python ranker.py` to generate rankings.
Uses a [generalized Plackett-Luce model](https://arxiv.org/pdf/2212.08543) to convert from seperate rankings into a global ranking.

## Limitations
We limit the number of scores per player to 100. This filtering uses the [pp](https://osu.ppy.sh/wiki/en/Performance_points) system which is unfortunate. There is another model in `models/ranker_complicated.stan` that may in theory handle different number of scores for different players, but it doesn't really work for various reasons.