from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
import numpy as np
import pickle
import sys

# Read parameters from shell call
# center or pre-border
region = sys.argv[1]
# range from 2 to 19
target_id = int(sys.argv[2])


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


# Real data
with open('input_data/resampled_td_914463.pkl', 'rb') as pkl_file:
    td = pickle.load(pkl_file)

if region == 'center':
    region_ix = [range(975, 1225), range(1975, 2225)]
elif region == 'pre-border':
    region_ix = [range(1350, 1600), range(2350, 2600)]
else:
    sys.exit()

data = stack_data(region_ix)

# b) Initialise analysis object and define settings
network_analysis = MultivariateTE()
settings = {'cmi_estimator': 'JidtKraskovCMI',
            'history_target': 1,
            'n_perm_max_stat': 100,
            'alpha_max_stat': 0.05,
            'permute_in_time': True,
            'max_lag_sources': 5,
            'min_lag_sources': 1,
            'add_conditionals': [(0, 1), (0, 2), (1, 1), (1, 2)]}

# c) Run analysis
sources = list(range(2, 20))
sources.remove(target_id)
res = network_analysis.analyse_single_target(settings=settings, data=data, target=target_id, sources=sources)

# Save results dictionary using pickle
path = 'target_results/{}_res.{}.pkl'.format(region, str(target_id))
pickle.dump(res, open(path, 'wb'))
