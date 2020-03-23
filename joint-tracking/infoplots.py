import jpype
import numpy as np
import pickle
import argparse
from collections import defaultdict


# flatten arrays
def flatten_array(arr):
    return [item for sublist in arr for item in sublist]


def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


def get_te(destinations, sources):
    avg_te = {}
    local_te = {}
    params = {'embedding': {}, 'delay': {}}

    if destinations == dict:
        for source in sources:
            avg_te[source] = {}
            local_te[source] = {}
            for destination in destinations:
                avg_te[source][destination] = []
                local_te[source][destination] = []

        for destination in destinations:
            params['embedding'][destination] = []
            params['delay'][destination] = []

        # create a TE calculator
        te_calc_class = jpype.JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
        te_calc = te_calc_class()

        for destination in destinations:
            # destination is data for all 8 neurons
            for i in range(8):
                # specify neuron time series
                dest_array = destinations[destination][:, i].tolist()
                # Since we're dealing with a well-defined simulation, we embed the destination only
                te_calc.setProperty(te_calc_class.PROP_AUTO_EMBED_METHOD,
                                    te_calc_class.AUTO_EMBED_METHOD_RAGWITZ_DEST_ONLY)
                te_calc.setProperty(te_calc_class.PROP_K_SEARCH_MAX, "6")
                te_calc.setProperty(te_calc_class.PROP_TAU_SEARCH_MAX, "6")
                # Supply source embedding
                te_calc.setProperty(te_calc_class.L_PROP_NAME, "1")
                te_calc.setProperty(te_calc_class.L_TAU_PROP_NAME, "1")

                # Check the auto-selected parameters and print out the result:
                optimised_k = int(te_calc.getProperty(te_calc_class.K_PROP_NAME))
                optimised_k_tau = int(te_calc.getProperty(te_calc_class.K_TAU_PROP_NAME))
                # optimisedL = int(teCalc.getProperty(teCalcClass.L_PROP_NAME))
                # optimisedLTau = int(teCalc.getProperty(teCalcClass.L_TAU_PROP_NAME))

                params['embedding'][destination].append(optimised_k)
                params['delay'][destination].append(optimised_k_tau)

                for source in sources:
                    source_array = sources[source]
                    # Since we're auto-embedding, no need to supply k, l, k_tau, l_tau here:
                    te_calc.initialise()

                    # Compute TE
                    te_calc.setObservations(source_array, dest_array)

                    te = te_calc.computeAverageLocalOfObservations()
                    local_measures = te_calc.computeLocalOfPreviousObservations()

                    avg_te[source][destination].append(te)
                    local_te[source][destination].append(list(local_measures))

    else:
        print('not a dict')
        for source in sources:
            avg_te[source] = []
            local_te[source] = []

        # create a TE calculator
        te_calc_class = jpype.JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
        te_calc = te_calc_class()

        dest_array = destinations
        # Since we're dealing with a well-defined simulation, we embed the destination only
        te_calc.setProperty(te_calc_class.PROP_AUTO_EMBED_METHOD,
                            te_calc_class.AUTO_EMBED_METHOD_RAGWITZ_DEST_ONLY)
        te_calc.setProperty(te_calc_class.PROP_K_SEARCH_MAX, "6")
        te_calc.setProperty(te_calc_class.PROP_TAU_SEARCH_MAX, "6")
        # Supply source embedding
        te_calc.setProperty(te_calc_class.L_PROP_NAME, "1")
        te_calc.setProperty(te_calc_class.L_TAU_PROP_NAME, "1")

        for source in sources:
            source_array = sources[source]
            # Since we're auto-embedding, no need to supply k, l, k_tau, l_tau here:
            te_calc.initialise()

            # Compute TE
            te_calc.setObservations(source_array, dest_array)

            te = te_calc.computeAverageLocalOfObservations()
            local_measures = te_calc.computeLocalOfPreviousObservations()

            avg_te[source].append(te)
            local_te[source].append(list(local_measures))

    return avg_te, local_te, params


def get_mi(destinations, sources):
    avg_mi = {}
    local_mi = {}

    for source in sources:
        avg_mi[source] = {}
        local_mi[source] = {}
        for destination in destinations:
            avg_mi[source][destination] = []
            local_mi[source][destination] = []

    mi_calc_class = jpype.JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov2
    mi_calc = mi_calc_class()

    for destination in destinations:
        # destination is data for all 8 neurons
        for i in range(8):
            # specify neuron time series
            dest_array = destinations[destination][:, i].tolist()

            mi_calc.setProperty("NORMALISE", "true")

            for source in sources:
                source_array = sources[source]

                mi_calc.initialise(1, 1)

                # Compute MI
                mi_calc.setObservations(source_array, dest_array)

                mi = mi_calc.computeAverageLocalOfObservations()
                local_measures = mi_calc.computeLocalOfPreviousObservations()

                avg_mi[source][destination].append(mi)
                local_mi[source][destination].append(list(local_measures))
    return avg_mi, local_mi


def get_ais(destinations):
    avg_ais = {}
    local_ais = {}
    params = {'embedding': {}, 'delay': {}}

    for destination in destinations:
        avg_ais[destination] = []
        local_ais[destination] = []
        params['embedding'][destination] = []
        params['delay'][destination] = []

    # 1. Construct the calculator:
    ais_calc_class = jpype.JPackage("infodynamics.measures.continuous.kraskov").ActiveInfoStorageCalculatorKraskov
    ais_calc = ais_calc_class()
    for destination in destinations:
        # destination is data for all 8 neurons
        for i in range(8):
            # specify neuron time series
            dest_array = destinations[destination][:, i].tolist()

            # 2. Set any properties to non-default values:
            ais_calc.setProperty("AUTO_EMBED_METHOD", "MAX_CORR_AIS")
            ais_calc.setProperty("AUTO_EMBED_K_SEARCH_MAX", "6")
            ais_calc.setProperty("AUTO_EMBED_TAU_SEARCH_MAX", "6")

            optimised_k = int(ais_calc.getProperty(ais_calc_class.K_PROP_NAME))
            optimised_k_tau = int(ais_calc.getProperty(ais_calc_class.TAU_PROP_NAME))

            params['embedding'][destination].append(optimised_k)
            params['delay'][destination].append(optimised_k_tau)

            # 3. Initialise the calculator for (re-)use:
            ais_calc.initialise()
            # 4. Supply the sample data:
            ais_calc.setObservations(dest_array)

            # 5. Compute the estimate:
            ais = ais_calc.computeAverageLocalOfObservations()
            local_measures = ais_calc.computeLocalOfPreviousObservations()

            avg_ais[destination].append(ais)
            local_ais[destination].append(list(local_measures))

    return avg_ais, local_ais, params


def get_cross_mi(action, brain, conditioning):
    all_lags = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # mi_calc_class = jpype.JPackage(
    # "infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov2
    mi_calc_class = jpype.JPackage(
        "infodynamics.measures.continuous.kraskov").ConditionalMutualInfoCalculatorMultiVariateKraskov2
    mi_calc = mi_calc_class()

    dest_array = action
    conditional_array = conditioning
    mi_calc.setProperty("NORMALISE", "true")

    for lag in range(1, 2):
        mi_calc.setProperty("PROP_TIME_DIFF", str(lag))

        for source in brain:
            for i in range(8):
                # specify neuron time series
                source_array = brain[source][:, i].tolist()
                mi_calc.initialise(1, 1, 1)

                # Compute MI
                mi_calc.setObservations(source_array, dest_array, conditional_array)

                mi = mi_calc.computeAverageLocalOfObservations()

                sig_distr = mi_calc.computeSignificance(1000)
                sig_mean = sig_distr.getMeanOfDistribution()
                sig_std = sig_distr.getStdOfDistribution()
                sig_p = sig_distr.pValue

                # print("Neuron %d, MI = %.4f bits (null: %.4f +/- %.4f std dev.); p(surrogate > measured)=%.5f "
                #       "from %d surrogates)" %
                #       (i, mi, sig_mean, sig_std, sig_p, 1000))

                local_measures = mi_calc.computeLocalOfPreviousObservations()

                all_lags[lag]['avg_mi'][source].append(mi)
                all_lags[lag]['local_mi'][source].append(list(local_measures))
                all_lags[lag]['sig_mi'][source].append((sig_mean, sig_std, sig_p))

    all_lags_regular = default_to_regular(all_lags)

    return all_lags_regular


def main(measure, trial_num):
    # jar location
    jarLocation = "/Users/katja/javawd/infodynamics-dist-1.4/infodynamics.jar"
    # Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
    jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

    # get data
    pkl_file = open('input_data/resampled_td_914463.pkl', 'rb')
    td = pickle.load(pkl_file)
    pkl_file.close()

    # pkl_file = open('td_immobile_914463.pkl', 'rb')
    # td = pickle.load(pkl_file)
    # pkl_file.close()
    # trial_num = 30  # target in position 10
    # td = td[0]  # tracker in position 0

    target_pos = flatten_array(td['target_pos'][trial_num].tolist())
    target_dist = flatten_array((td['target_pos'][trial_num] - td['tracker_pos'][trial_num]).tolist())
    left_border_dist = flatten_array((-20 - td['tracker_pos'][trial_num]).tolist())
    right_border_dist = flatten_array((20 - td['tracker_pos'][trial_num]).tolist())

    outputs_a1 = td['output_a1'][trial_num]
    brain_states_a1 = td['brain_state_a1'][trial_num]

    outputs_a2 = td['output_a2'][trial_num]
    brain_states_a2 = td['brain_state_a2'][trial_num]

    left_motor = td['keypress'][trial_num][:, 0]
    right_motor = td['keypress'][trial_num][:, 1]

    brain_a1 = {'outputs_a1': outputs_a1, 'activation_a1': brain_states_a1}
    brain_a2 = {'outputs_a2': outputs_a2, 'activation_a2': brain_states_a2}

    destinations_data = {'outputs_a1': outputs_a1, 'outputs_a2': outputs_a2,
                         'activation_a1': brain_states_a1, 'activation_a2': brain_states_a2}

    sources_data = {'target_dist': target_dist, 'left_border_dist': left_border_dist,
                    'right_border_dist': right_border_dist}

    if measure == "te":
        sources_data = {'target_dist': target_dist, 'left_border_dist': left_border_dist,
                        'right_border_dist': right_border_dist,
                        'left_motor': left_motor, 'right_motor': right_motor}

        # computed_avg_te, computed_local_te, computed_params = get_te(destinations_data, sources_data)
        #
        # output = open('{}_trial{}.pkl'.format(measure, trial_num+1), 'wb')
        # pickle.dump((computed_avg_te, computed_local_te, computed_params), output)
        # output.close()
        #
        # print('done stimuli to neurons')

        sources_data_exit_a1 = {}
        for source in brain_a1:
            for i in range(8):
                k = source + '_n' + str(i)
                sources_data_exit_a1[k] = brain_a1[source][:, i]

        destinations_data_exit_a1 = left_motor.tolist()
        print(left_motor)
        print(type(left_motor))

        print(sources_data_exit_a1.keys())
        computed_avg_te, computed_local_te, computed_params = get_te(destinations_data_exit_a1, sources_data_exit_a1)

        output = open('{}_trial{}_exit1.pkl'.format(measure, trial_num+1), 'wb')
        pickle.dump((computed_avg_te, computed_local_te, computed_params), output)
        output.close()

        sources_data_exit_a2 = {}
        for source in brain_a2:
            for i in range(8):
                k = source + '_n' + str(i)
                sources_data_exit_a2[k] = brain_a2[source][:, i]

        destinations_data_exit_a2 = right_motor.tolist()

        computed_avg_te, computed_local_te, computed_params = get_te(destinations_data_exit_a2, sources_data_exit_a2)

        output = open('{}_trial{}_exit2.pkl'.format(measure, trial_num+1), 'wb')
        pickle.dump((computed_avg_te, computed_local_te, computed_params), output)
        output.close()

    elif measure == "mi":
        computed_avg_mi, computed_local_mi = get_mi(destinations_data, sources_data)

        output = open('{}_trial{}.pkl'.format(measure, trial_num + 1), 'wb')
        pickle.dump((computed_avg_mi, computed_local_mi), output)
        output.close()

    elif measure == "ais":
        computed_avg_ais, computed_local_ais, computed_params = get_ais(destinations_data)

        output = open('{}_trial{}.pkl'.format(measure, trial_num + 1), 'wb')
        pickle.dump((computed_avg_ais, computed_local_ais, computed_params), output)
        output.close()

    elif measure == "cross_mi":
        brain2_to_action1 = get_cross_mi(left_motor, brain_a2, target_pos)
        brain1_to_action2 = get_cross_mi(right_motor, brain_a1, target_pos)

        output = open('{}_trial{}.pkl'.format(measure, trial_num + 1), 'wb')
        pickle.dump((brain2_to_action1, brain1_to_action2), output)
        output.close()

    jpype.shutdownJVM()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("measure", type=str)
    parser.add_argument("trial_num", type=int)
    args = parser.parse_args()
    main(args.measure, args.trial_num)
