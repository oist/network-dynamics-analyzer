import pickle
from idtxl.stats import network_fdr

# Set target numbers to combine
targets = list(range(2, 20))

# Load results using pickle
res_list = []
for target_id in targets:
    path = 'target_results/res.{}.pkl'.format(str(target_id))
    res_list.append(pickle.load(open(path, 'rb')))

# with FDR correction for multiple comparisons
res = network_fdr({'alpha_fdr': 0.05}, *res_list)

# # without FDR correction for multiple comparisons
# res = res_list[0]
# res.combine_results(*res_list[1:])
