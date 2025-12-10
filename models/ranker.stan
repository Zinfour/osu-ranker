functions {
    real gpl_lpmf(array[] int y, array[] int s, vector theta) {
        vector[size(y)] theta_y = theta[y];

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
                    array[] int score_to_player,
                    array[] int score_ties,
                    vector beta,
                    array[] int scores_per_map,
                    array[] int rt) {
        real targt = 0;
        
        for (m in m_slice) {
            targt += gpl_lpmf(score_to_player[rt[m]+1:rt[m]+scores_per_map[m]] | score_ties[rt[m]+1:rt[m]+scores_per_map[m]], beta);
        }

        return targt;
    }
}
data {
    int<lower=1> M; // # Maps x Mods
    int<lower=1> S; // # scores
    int<lower=1> P; // # players
    array[M] int<lower=1, upper=S> scores_per_map;
    array[S] int<lower=1, upper=P> score_to_player;
    array[S] int<lower=0, upper=1> score_ties;
}
transformed data {
    array[M] int rt;
    int j = 0;
    for (m in 1:M) {
        rt[m] = j;
        j += scores_per_map[m];
    }
}

parameters {
    vector<lower=0, upper=1>[P] beta; // player ratings
    real<lower=0> alpha_param;
    real<lower=1> beta_param;
}
model {
    beta ~ beta(alpha_param, beta_param);
    int grainsize = 1;
    target += reduce_sum(partial_sum, linspaced_int_array(M, 1, M), grainsize,
                        score_to_player,
                        score_ties,
                        beta,
                        scores_per_map,
                        rt);

}


