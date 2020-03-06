"""Provide unit tests for high-level PID estimation."""
import pytest
import numpy as np
from idtxl.partial_information_decomposition import (
                                        PartialInformationDecomposition)
from idtxl.data import Data
from test_estimators_pid import optimiser_missing, float128_not_available


@optimiser_missing
def test_pid_user_input():
    """Test if user input is handled correctly."""
    # Test missing estimator name
    pid = PartialInformationDecomposition()
    with pytest.raises(RuntimeError):
        pid.analyse_single_target(settings={}, data=Data(), target=0,
                                  sources=[1, 2])

    # Test wrong estimator name
    settings = {'pid_estimator': 'TestPID'}
    with pytest.raises(RuntimeError):
        pid.analyse_single_target(settings=settings, data=Data(), target=0,
                                  sources=[1, 2])

    # Test default lags for network_analysis
    settings = {'pid_estimator': 'TartuPID'}
    data = Data(np.random.randint(0, 10, size=(5, 100)), dim_order='ps',
                normalise=False)
    results = pid.analyse_network(settings=settings, data=data,
                                  targets=[0, 1, 2],
                                  sources=[[1, 3], [2, 4], [0, 1]])
    assert np.all(results.settings['lags_pid'] == [1, 1]), (
        'Lags were not set to default.')

    n = 20
    alph = 2
    x = np.random.randint(0, alph, n)
    y = np.random.randint(0, alph, n)
    z = np.logical_xor(x, y).astype(int)
    data = Data(np.vstack((x, y, z)), 'ps', normalise=False)

    # Test two-tailed significance test
    settings = {'pid_estimator': 'TartuPID', 'tail': 'two', 'lags_pid': [0, 0]}
    pid = PartialInformationDecomposition()

    with pytest.raises(RuntimeError):  # Test incorrect number of sources
        pid.analyse_single_target(settings=settings, data=data, target=2,
                                  sources=[1, 2, 3])
    settings['lags_pid'] = [0, 0, 0]
    with pytest.raises(RuntimeError):  # Test incorrect number of lags
        pid.analyse_single_target(settings=settings, data=data, target=2,
                                  sources=[1, 3])
    settings['lags_pid'] = [n * 3, 0]
    with pytest.raises(RuntimeError):  # Test lag > no. samples
        pid.analyse_single_target(settings=settings, data=data, target=2,
                                  sources=[0, 1])
    settings['lags_pid'] = [n, 0]
    with pytest.raises(RuntimeError):  # Test lag == no. samples
        pid.analyse_single_target(settings=settings, data=data, target=2,
                                  sources=[0, 1])
    settings['lags_pid'] = [0, 0]
    with pytest.raises(RuntimeError):  # Test target in sources
        pid.analyse_single_target(settings=settings, data=data, target=2,
                                  sources=[2, 3])
    with pytest.raises(IndexError):  # Test target not in processes
        pid.analyse_single_target(settings=settings, data=data, target=5,
                                  sources=[0, 1])


@optimiser_missing
@float128_not_available
def test_network_analysis():
    """Test call to network_analysis method."""
    n = 100
    alph = 2
    x = np.random.randint(0, alph, n)
    y = np.random.randint(0, alph, n)
    z = np.logical_xor(x, y).astype(int)
    data = Data(np.vstack((x, y, z)), 'ps', normalise=False)

    # Run Tartu estimator
    pid = PartialInformationDecomposition()
    settings = {'pid_estimator': 'TartuPID', 'tail': 'two',
                'lags_pid': [[0, 0], [0, 0]]}
    est_tartu = pid.analyse_network(settings=settings,
                                    data=data, targets=[0, 2],
                                    sources=[[1, 2], [0, 1]])
    assert 0.9 < est_tartu._single_target[2]['syn_s1_s2'] <= 1.1, (
        'Tartu estimator incorrect synergy: {0}, should approx. 1'.format(
                                    est_tartu._single_target[2]['syn_s1_s2']))

    # Run Sydney estimator
    settings = {'pid_estimator': 'SydneyPID',
                'alph_s1': alph,
                'alph_s2': alph,
                'alph_t': alph,
                'max_unsuc_swaps_row_parm': 60,
                'num_reps': 63,
                'max_iters': 1000,
                'pid_estimator': 'SydneyPID',
                'lags_pid': [[0, 0], [0, 0]]}
    est_sydney = pid.analyse_network(settings=settings,
                                     data=data, targets=[0, 2],
                                     sources=[[1, 2], [0, 1]])
    assert 0.9 < est_sydney._single_target[2]['syn_s1_s2'] <= 1.1, (
        'Sydney estimator incorrect synergy: {0}, should approx. 1'.format(
                                    est_tartu._single_target[2]['syn_s1_s2']))


@optimiser_missing
@float128_not_available
def test_analyse_single_target():
    """Test call to network_analysis method."""
    n = 100
    alph = 2
    x = np.random.randint(0, alph, n)
    y = np.random.randint(0, alph, n)
    z = np.logical_xor(x, y).astype(int)
    data = Data(np.vstack((x, y, z)), 'ps', normalise=False)

    # Run Tartu estimator
    pid = PartialInformationDecomposition()
    settings = {'pid_estimator': 'TartuPID',
                'tail': 'two',
                'lags_pid': [0, 0]}
    est_tartu = pid.analyse_single_target(settings=settings, data=data,
                                          target=2, sources=[0, 1])
    assert 0.9 < est_tartu._single_target[2]['syn_s1_s2'] <= 1.1, (
        'Tartu estimator incorrect synergy: {0}, should approx. 1'.format(
            est_tartu._single_target[2]['syn_s1_s2']))
    assert est_tartu._single_target[2]['unq_s1'] < 0.1, (
        'Tartu estimator incorrect unique s1: {0}, should approx. 0'.format(
            est_tartu._single_target[2]['unq_s1']))
    assert est_tartu._single_target[2]['unq_s2'] < 0.1, (
        'Tartu estimator incorrect unique s2: {0}, should approx. 0'.format(
            est_tartu._single_target[2]['unq_s2']))

    # Run Sydney estimator
    settings = {'pid_estimator': 'SydneyPID',
                'alph_s1': alph,
                'alph_s2': alph,
                'alph_t': alph,
                'max_unsuc_swaps_row_parm': 60,
                'num_reps': 63,
                'max_iters': 1000,
                'pid_estimator': 'SydneyPID',
                'lags_pid': [0, 0]}
    est_sydney = pid.analyse_single_target(settings=settings, data=data,
                                           target=2, sources=[0, 1])
    assert 0.9 < est_sydney._single_target[2]['syn_s1_s2'] <= 1.1, (
        'Sydney estimator incorrect synergy: {0}, should approx. 1'.format(
            est_sydney._single_target[2]['syn_s1_s2']))
    assert est_sydney._single_target[2]['unq_s1'] < 0.1, (
        'Sydney estimator incorrect unique s1: {0}, should approx. 0'.format(
            est_sydney._single_target[2]['unq_s1']))
    assert est_sydney._single_target[2]['unq_s2'] < 0.1, (
        'Sydney estimator incorrect unique s2: {0}, should approx. 0'.format(
            est_sydney._single_target[2]['unq_s2']))

if __name__ == '__main__':
    test_pid_user_input()
    test_network_analysis()
    test_analyse_single_target()
