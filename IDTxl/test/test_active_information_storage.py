"""Test AIS analysis class.

This module provides unit tests for the AIS analysis class.
"""
import pytest
import random as rn
import numpy as np
from idtxl.data import Data
from idtxl.active_information_storage import ActiveInformationStorage
from idtxl.estimators_jidt import JidtDiscreteCMI
from test_estimators_jidt import jpype_missing

package_missing = False
try:
    import pyopencl
except ImportError as err:
    package_missing = True
opencl_missing = pytest.mark.skipif(
    package_missing,
    reason="Jpype is missing, JIDT estimators are not available")


@jpype_missing
def test_return_local_values():
    """Test estimation of local values."""
    max_lag = 5
    settings = {
        'cmi_estimator': 'JidtKraskovCMI',
        'local_values': True,  # request calculation of local values
        'n_perm_max_stat': 21,
        'n_perm_min_stat': 21,
        'n_perm_mi': 21,
        'max_lag': max_lag,
        'tau': 1}
    data = Data()
    data.generate_mute_data(100, 3)
    ais = ActiveInformationStorage()
    processes = [1, 2]
    results = ais.analyse_network(settings, data, processes)

    for p in processes:
        lais = results.get_single_process(p, fdr=False)['ais']
        if lais is np.nan:
            continue
        assert type(lais) is np.ndarray, (
            'LAIS estimation did not return an array of values: {0}'.format(
                lais))
        assert lais.shape[0] == data.n_replications, (
            'Wrong dim (no. replications) in LAIS estimate: {0}'.format(
                lais.shape))
        assert lais.shape[1] == data.n_realisations_samples((0, max_lag)), (
            'Wrong dim (no. samples) in LAIS estimate: {0}'.format(lais.shape))


@jpype_missing
def test_ActiveInformationStorage_init():
    """Test instance creation for ActiveInformationStorage class."""
    # Test error on missing estimator
    settings = {'max_lag': 5}
    data = Data()
    data.generate_mute_data(10, 3)
    ais = ActiveInformationStorage()
    with pytest.raises(RuntimeError):
        ais.analyse_single_process(settings, data, process=0)

    # Test tau larger than maximum lag
    settings['cmi_estimator'] = 'JidtKraskovCMI'
    settings['tau'] = 10
    with pytest.raises(RuntimeError):
        ais.analyse_single_process(settings, data, process=0)
    # Test negative tau and maximum lag
    settings['tau'] = -10
    with pytest.raises(RuntimeError):
        ais.analyse_single_process(settings, data, process=0)
    settings['tau'] = 1
    settings['max_lag'] = -5
    with pytest.raises(RuntimeError):
        ais.analyse_single_process(settings, data, process=0)

    # Invalid: process is not an int
    settings['max_lag'] = 5
    with pytest.raises(RuntimeError):  # no int
        ais.analyse_single_process(settings, data, process=1.5)
    with pytest.raises(RuntimeError):  # negative
        ais.analyse_single_process(settings, data, process=-1)
    with pytest.raises(RuntimeError):  # not in data
        ais.analyse_single_process(settings, data, process=10)
    with pytest.raises(RuntimeError):  # wrong type
        ais.analyse_single_process(settings, data, process={})


@jpype_missing
def test_add_conditional_manually():
    """Enforce the conditioning on additional variables."""
    settings = {'cmi_estimator': 'JidtKraskovCMI',
                'max_lag': 5,
                'n_perm_max_stat': 21,
                'n_perm_min_stat': 21,
                'n_perm_mi': 21}
    data = Data()
    data.generate_mute_data(10, 3)
    ais = ActiveInformationStorage()

    # Add a conditional with a lag bigger than the max_lag requested above
    settings['add_conditionals'] = (8, 0)
    with pytest.raises(IndexError):
        ais.analyse_single_process(settings=settings, data=data, process=0)

    # Add valid conditionals and test if they were added
    settings['add_conditionals'] = [(0, 1), (1, 3)]
    ais._initialise(settings, data, 0)
    # Get list of conditionals after intialisation and convert absolute samples
    # back to lags for comparison.
    cond_list = ais._idx_to_lag(ais.selected_vars_full)
    assert settings['add_conditionals'][0] in cond_list, (
        'First enforced conditional is missing from results.')
    assert settings['add_conditionals'][1] in cond_list, (
        'Second enforced conditional is missing from results.')


def test_analyse_network():
    """Test AIS estimation for the whole network."""
    settings = {
        'cmi_estimator': 'JidtKraskovCMI',
        'n_perm_max_stat': 21,
        'n_perm_min_stat': 21,
        'n_perm_mi': 21,
        'max_lag': 5,
        'tau': 1}
    data = Data()
    data.generate_mute_data(10, 3)
    ais = ActiveInformationStorage()
    # Test analysis of 'all' processes
    results = ais.analyse_network(settings, data)
    k = results.processes_analysed
    assert all(np.array(k) == np.arange(data.n_processes)), (
                'Network analysis did not run on all targets.')
    # Test check for correct definition of processes
    with pytest.raises(ValueError):  # no list
        ais.analyse_network(settings, data=data, processes={})
    with pytest.raises(ValueError):  # no list of ints
        ais.analyse_network(settings, data=data, processes=[1.5, 0.7])


@jpype_missing
def test_single_source_storage_gaussian():
    n = 1000
    proc_1 = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    proc_2 = [rn.normalvariate(0, 1) for r in range(n)]  # correlated src
    # Cast everything to numpy so the idtxl estimator understands it.
    data = Data(np.array([proc_1, proc_2]), dim_order='ps')
    settings = {
        'cmi_estimator': 'JidtKraskovCMI',
        'alpha_mi': 0.05,
        'tail_mi': 'one_bigger',
        'n_perm_max_stat': 21,
        'n_perm_min_stat': 21,
        'n_perm_mi': 21,
        'max_lag': 5,
        'tau': 1
        }
    processes = [1]
    network_analysis = ActiveInformationStorage()
    results = network_analysis.analyse_network(settings, data, processes)
    print('AIS for random normal data without memory (expected is NaN): '
          '{0}'.format(results._single_process[1].ais))
    assert results._single_process[1].ais is np.nan, (
        'Estimator did not return nan for memoryless data.')


@jpype_missing
@opencl_missing
def test_compare_jidt_open_cl_estimator():
    """Compare results from OpenCl and JIDT estimators for AIS calculation."""
    data = Data()
    data.generate_mute_data(1000, 2)
    settings = {
        'cmi_estimator': 'OpenCLKraskovCMI',
        'alpha_mi': 0.05,
        'tail_mi': 'one_bigger',
        'n_perm_max_stat': 21,
        'n_perm_min_stat': 21,
        'n_perm_mi': 21,
        'max_lag': 5,
        'tau': 1
        }
    processes = [2, 3]
    network_analysis = ActiveInformationStorage()
    res_opencl = network_analysis.analyse_network(settings, data, processes)
    settings['cmi_estimator'] = 'JidtKraskovCMI'
    res_jidt = network_analysis.analyse_network(settings, data, processes)
    # Note that I require equality up to three digits. Results become more
    # exact for bigger data sizes, but this takes too long for a unit test.
    ais_opencl_2 = res_opencl._single_process[2].ais
    ais_jidt_2 = res_jidt._single_process[2].ais
    ais_opencl_3 = res_opencl._single_process[3].ais
    ais_jidt_3 = res_jidt._single_process[3].ais
    print('AIS for MUTE data proc 2 - opencl: {0} and jidt: {1}'.format(
        ais_opencl_2, ais_jidt_2))
    print('AIS for MUTE data proc 3 - opencl: {0} and jidt: {1}'.format(
        ais_opencl_3, ais_jidt_3))
    if not (ais_opencl_2 is np.nan or ais_jidt_2 is np.nan):
        assert (ais_opencl_2 - ais_jidt_2) < 0.05, (
            'AIS results differ between OpenCl and JIDT estimator.')
    else:
        assert ais_opencl_2 is ais_jidt_2, (
            'AIS results differ between OpenCl and JIDT estimator.')
    if not (ais_opencl_3 is np.nan or ais_jidt_3 is np.nan):
        assert (ais_opencl_3 - ais_jidt_3) < 0.05, (
            'AIS results differ between OpenCl and JIDT estimator.')
    else:
        assert ais_opencl_3 is ais_jidt_3, (
            'AIS results differ between OpenCl and JIDT estimator.')


def test_discrete_input():
    """Test AIS estimation from discrete data."""
    # Generate AR data
    order = 1
    n = 10000 - order
    self_coupling = 0.5
    process = np.zeros(n + order)
    process[0:order] = np.random.normal(size=(order))
    for n in range(order, n + order):
        process[n] = self_coupling * process[n - 1] + np.random.normal()

    # Discretise data
    settings = {'discretise_method': 'equal',
                'n_discrete_bins': 5}
    est = JidtDiscreteCMI(settings)
    process_dis, temp = est._discretise_vars(var1=process, var2=process)
    data = Data(process_dis, dim_order='s', normalise=False)
    settings = {
        'cmi_estimator': 'JidtDiscreteCMI',
        'discretise_method': 'none',
        'n_discrete_bins': 5,  # alphabet size of the variables
        'n_perm_max_stat': 21,
        'n_perm_min_stat': 21,
        'n_perm_mi': 21,
        'max_lag': 2}
    nw = ActiveInformationStorage()
    nw.analyse_single_process(settings=settings, data=data, process=0)


@jpype_missing
def test_define_candidates():
    """Test candidate definition from a list of procs and a list of samples."""
    target = 1
    tau_target = 3
    max_lag_target = 10
    current_val = (target, 10)
    procs = [target]
    samples = np.arange(current_val[1] - 1, current_val[1] - max_lag_target,
                        -tau_target)
    # Test if candidates that are added manually to the conditioning set are
    # removed from the candidate set.
    nw = ActiveInformationStorage()
    nw.current_value = current_val
    settings = [
        {'add_conditionals': None},
        {'add_conditionals': (2, 3)},
        {'add_conditionals': [(2, 3), (4, 1)]},
        {'add_conditionals': [(1, 9)]},
        {'add_conditionals': [(1, 9), (2, 3), (4, 1)]}]
    for s in settings:
        nw.settings = s
        candidates = nw._define_candidates(procs, samples)
        assert (1, 9) in candidates, 'Sample missing from candidates: (1, 9).'
        assert (1, 6) in candidates, 'Sample missing from candidates: (1, 6).'
        assert (1, 3) in candidates, 'Sample missing from candidates: (1, 3).'
        if s['add_conditionals'] is not None:
            if type(s['add_conditionals']) is tuple:
                cond_ind = nw._lag_to_idx([s['add_conditionals']])
            else:
                cond_ind = nw._lag_to_idx(s['add_conditionals'])
            for c in cond_ind:
                assert c not in candidates, (
                    'Sample added erronously to candidates: {}.'.format(c))


if __name__ == '__main__':
    test_define_candidates()
    test_return_local_values()
    test_discrete_input()
    test_analyse_network()
    test_ActiveInformationStorage_init()
    test_single_source_storage_gaussian()
    test_compare_jidt_open_cl_estimator()
