import pickle
from idtxl.stats import network_fdr
import sys
import os

# Set target numbers to combine
region = sys.argv[1]
# range from 2 to 19
targets = list(range(2, 20))

# Load results using pickle
res_list = []
for target_id in targets:
    path = 'target_results/{}_res.{}.pkl'.format(region, str(target_id))
    if os.path.exists(path):
        res_list.append(pickle.load(open(path, 'rb')))

# # with FDR correction for multiple comparisons
# res = network_fdr({'alpha_fdr': 0.05}, *res_list)
#
# without FDR correction for multiple comparisons
res = res_list[0]
res.combine_results(*res_list[1:])

with open('target_results/network_{}_full.p'.format(region), 'wb') as pkl:
    pickle.dump(res, pkl)
