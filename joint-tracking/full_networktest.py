from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
import numpy as np
import pickle
from idtxl.network_comparison import NetworkComparison

# Real data
with open('resampled_td_914463.pkl', 'rb') as pkl_file:
    td = pickle.load(pkl_file)


"""
before border: 400-600 1400-1600 2400-2600
after border: 600-800 1600-1800 2600-2800
around middle: 950-1250 1950-2250
"""
# pre_border_ix = np.array([np.arange(400, 600), np.arange(1400, 1600), np.arange(2400, 2600)]).reshape(600)
# after_border_ix = np.array([np.arange(600, 800), np.arange(1600, 1800), np.arange(2600, 2800)]).reshape(600)
# center_ix = np.array([np.arange(950, 1250), np.arange(1950, 2250)]).reshape(600)

pre_border_ix = [range(1350, 1600), range(2350, 2600)]
# after_border_ix = np.array([np.arange(600, 800), np.arange(1600, 1800), np.arange(2600, 2800)]).reshape(600)
center_ix = [range(975, 1225), range(1975, 2225)]


def stack_data(ixs):
    all_trials = None
    starting = True
    for i in range(6):
        for ix in ixs:
            activations_a1 = td['brain_state_a1'][i][ix]
            activations_a2 = td['brain_state_a2'][i][ix]
            left_motor = td['keypress'][i][ix, 0].tolist()
            right_motor = td['keypress'][i][ix, 1].tolist()
            target_pos = td['target_pos'][i][ix]
            tracker_pos = td['tracker_pos'][i][ix]
            all_center = np.column_stack((target_pos, tracker_pos, activations_a1, activations_a2,
                                          right_motor, left_motor))
            if starting:
                all_trials = all_center
                starting = False
            else:
                all_trials = np.dstack((all_trials, all_center))
    stacked_data = Data(all_trials, dim_order='spr')
    return stacked_data


data_center = stack_data(center_ix)
data_pre_border = stack_data(pre_border_ix)

# b) Initialise analysis object and define settings
network_analysis = MultivariateTE()

# estimator choice: for non-linear continuous data use Kraskov:
# JidtKraskovAIS, JidtKraskovCMI, JidtKraskovMI, JidtKraskovTE

# statistical testing - shuffle over time for a max of 200 permutations with standard alpha
# conditioning: (process index, lag)

settings = {'cmi_estimator': 'JidtKraskovCMI',
            'history_target': 1,
            'n_perm_max_stat': 100,
            'alpha_max_stat': 0.05,
            'permute_in_time': True,
            'max_lag_sources': 5,
            'min_lag_sources': 1,
            'add_conditionals': [(0, 1), (0, 2), (1, 1), (1, 2)]}

# c) Run analysis
center_res = network_analysis.analyse_network(settings=settings, data=data_center)
pre_border_res = network_analysis.analyse_network(settings=settings, data=data_pre_border)

with open('network_center_full.p', 'wb') as pklc:
    pickle.dump(center_res, pklc)
with open('network_preborder_full.p', 'wb') as pklb:
    pickle.dump(pre_border_res, pklb)


# compare
# center_res = pickle.load(open('network_center_full.p', 'rb'))
# pre_border_res = pickle.load(open('network_preborder_full.p', 'rb'))
#
# comparison_settings = {
#     'stats_type': 'independent',
#     'cmi_estimator': 'JidtKraskovCMI',
#     'tail_comp': 'two',
#     'n_perm_comp': 100,
#     'alpha_comp': 0.05,
#     'verbose': True
# }
#
# comparison = NetworkComparison()
#
# comparison_res = comparison.compare_within(data_a=data_center, data_b=data_pre_border,
#                           network_a=center_res, network_b=pre_border_res,
#                           settings=comparison_settings)
#
# print(comparison_res['sign'])
# pickle.dump(comparison_res, open('network_comparison_dict.p', 'wb'))
