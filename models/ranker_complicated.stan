functions {
    real gpl_lpmf(array[] int y, array[] int s, matrix theta, vector map_type, vector weights) {
        vector[size(y)] theta_y = 1 - exp(log(1 - (theta[y] * map_type)) ./ weights);

        vector[size(y)-1] summands;
        vector[size(y)] log1m_theta_y = log1m(theta_y);
        vector[size(y)] pre_summed = sum(log1m_theta_y) - cumulative_sum(log1m_theta_y);

        for (j in 1:size(y)-1) {
            if (s[j]) {
                summands[j] = log(theta_y[j]) + log1m_exp(pre_summed[j]) - log1m_exp(log1m_theta_y[j] + pre_summed[j]);
            } else {
                summands[j] = log(theta_y[j]) + pre_summed[j] - log1m_exp(log1m_theta_y[j] + pre_summed[j]);
            }
        }

        return sum(summands);
    }
    real partial_sum(array[] int m_slice,
                    int start, int end,
                    array[] vector map_types,
                    array[] int score_to_player,
                    array[] int score_ties,
                    matrix beta,
                    vector score_weights,
                    array[] int scores_per_map,
                    vector dirichlet_alpha,
                    array[] int rt2) {
        real targt = 0;
        
        for (m in m_slice) {
            targt += dirichlet_lpdf(map_types[m] | dirichlet_alpha);
            targt += gpl_lpmf(score_to_player[rt2[m]+1:rt2[m]+scores_per_map[m]] | score_ties[rt2[m]+1:rt2[m]+scores_per_map[m]], beta, map_types[m], score_weights[rt2[m]+1:rt2[m]+scores_per_map[m]]);
        }

        return targt;
    }
}
data {
    int<lower=1> M; // # Maps x Mods
    int<lower=1> S; // # scores
    int<lower=1> P; // # players
    int<lower=1> SKILL_TYPES;
    array[M] int<lower=1, upper=S> scores_per_map;
    array[S] int<lower=1, upper=P> score_to_player;
    array[S] int<lower=0, upper=1> score_ties;

    array[P] int<lower=1, upper=S> scores_per_player;
    array[S] int<lower=1, upper=S> player_to_scores;
}
transformed data {
    array[P] int rt1;
    int i = 0;
    for (p in 1:P) {
        rt1[p] = i;
        i += scores_per_player[p];
    }
    array[M] int rt2;
    int j = 0;
    for (m in 1:M) {
        rt2[m] = j;
        j += scores_per_map[m];
    }
}

parameters {
    matrix<lower=0, upper=1>[P, SKILL_TYPES] beta; // player ratings
    array[M] simplex[SKILL_TYPES] map_types;
    real<lower=1> alpha_param;
    real<lower=1> beta_param;
    vector<lower=0>[SKILL_TYPES] dirichlet_alpha;
    real<lower=0, upper=100> dirichlet_alpha2;
    vector[S-P] score_weights_raw;
}
model {
    vector[S] score_weights;
    {
        for (p in 1:P) {
            vector[scores_per_player[p]] tmp = simplex_jacobian(score_weights_raw[rt1[p] + 2 - p:rt1[p]+scores_per_player[p] - p]);
            tmp ~ dirichlet(rep_vector(dirichlet_alpha2, scores_per_player[p]));
            score_weights[player_to_scores[rt1[p]+1:rt1[p]+scores_per_player[p]]] = tmp/max(tmp);
        }
    }
    for (p in 1:P) {
        beta[p] ~ beta(alpha_param, beta_param);
    }
    // for (m in 1:M) {
    //     map_types[m] ~ dirichlet(dirichlet_alpha);
    //     score_to_player[rt2[m]+1:rt2[m]+scores_per_map[m]] ~ gpl(score_ties[rt2[m]+1:rt2[m]+scores_per_map[m]], beta, map_types[m], score_weights[rt2[m]+1:rt2[m]+scores_per_map[m]]);
    // }
    int grainsize = 1;
    target += reduce_sum(partial_sum, linspaced_int_array(M, 1, M), grainsize, map_types,
                        score_to_player,
                        score_ties,
                        beta,
                        score_weights,
                        scores_per_map,
                        dirichlet_alpha,
                        rt2);

}


