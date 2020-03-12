import pickle
from idtxl.visualise_graph import plot_network
import matplotlib.pyplot as plt


with open('target_results/network_center_full.p', 'rb') as pkl_file:
    center_res = pickle.load(pkl_file)
# d) Plot inferred network to console and via matplotlib
center_res.print_edge_list(weights='max_te_lag', fdr=False)
plot_network(results=center_res, weights='max_te_lag', fdr=False)

# with open('network_preborder_full.p', 'rb') as pkl_file:
#     pre_border_res = pickle.load(pkl_file)
# pre_border_res.print_edge_list(weights='max_te_lag', fdr=False)
# plot_network(results=pre_border_res, weights='max_te_lag', fdr=False)

plt.show()
