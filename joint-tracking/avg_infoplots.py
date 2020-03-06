import jpype
import pickle
import numpy as np


# flatten arrays
def flatten_array(arr):
    return [item for sublist in arr for item in sublist]


def get_te(destinations, sources):
    avg_te = {}
    local_te = {}
    params = {'embedding': {}, 'delay': {}}

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

            # # For auto-embedding of both source and destination:
            # #  a. Auto-embedding method
            # teCalc.setProperty(teCalcClass.PROP_AUTO_EMBED_METHOD, teCalcClass.AUTO_EMBED_METHOD_RAGWITZ)
            # #  b. Search range for embedding dimension (k) and delay (tau)
            # teCalc.setProperty(teCalcClass.PROP_K_SEARCH_MAX, "6")
            # teCalc.setProperty(teCalcClass.PROP_TAU_SEARCH_MAX, "6")

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


def get_cross_mi(action, brain):
    avg_mi = {}
    local_mi = {}

    for source in brain:
        avg_mi[source] = []
        local_mi[source] = []

    mi_calc_class = jpype.JPackage("infodynamics.measures.continuous.kraskov").MutualInfoCalculatorMultiVariateKraskov2
    mi_calc = mi_calc_class()

    dest_array = action
    mi_calc.setProperty("NORMALISE", "true")

    for source in brain:
        for i in range(8):
            # specify neuron time series
            source_array = brain[source][:, i].tolist()
            mi_calc.initialise(1, 1)

            # Compute MI
            mi_calc.setObservations(source_array, dest_array)

            mi = mi_calc.computeAverageLocalOfObservations()
            local_measures = mi_calc.computeLocalOfPreviousObservations()

            avg_mi[source].append(mi)
            local_mi[source].append(list(local_measures))

            # TODO: EmpiricalMeasurementDistribution = teCalc.computeSignificance(1000)
    return avg_mi, local_mi


def get_avg_measures(trial_data, measure):
    # num_trials = len(trial_data['target_pos'])
    num_trials = 3
    summed_global = 0
    summed_local = 0

    for trial_num in range(num_trials):
        print("Analyzing trial " + str(trial_num))
        target_dist = flatten_array((trial_data['target_pos'][trial_num] - trial_data['tracker_pos'][trial_num]).tolist())
        left_border_dist = flatten_array((-20 - trial_data['tracker_pos'][trial_num]).tolist())
        right_border_dist = flatten_array((20 - trial_data['tracker_pos'][trial_num]).tolist())

        outputs_a1 = trial_data['output_a1'][trial_num]
        brain_states_a1 = trial_data['brain_state_a1'][trial_num]

        outputs_a2 = trial_data['output_a2'][trial_num]
        brain_states_a2 = trial_data['brain_state_a2'][trial_num]

        destinations_data = {'outputs_a1': outputs_a1, 'outputs_a2': outputs_a2,
                             'activation_a1': brain_states_a1, 'activation_a2': brain_states_a2}

        sources_data = {'target_dist': target_dist, 'left_border_dist': left_border_dist,
                        'right_border_dist': right_border_dist}

        if measure == "te":
            computed_global, computed_local, _ = get_te(destinations_data, sources_data)

            if trial_num == 0:
                summed_global = computed_global
                summed_local = computed_local
                for source in summed_local:
                    for destination in summed_local[source]:
                        summed_global[source][destination] = np.array(computed_global[source][destination])
                        summed_local[source][destination] = np.array(computed_local[source][destination])
            else:
                for source in summed_local:
                    for destination in summed_local[source]:
                        summed_global[source][destination] = summed_global[source][destination] + \
                                                             np.array(computed_global[source][destination])
                        summed_local[source][destination] = summed_local[source][destination] + \
                                                            np.array(computed_local[source][destination])

        elif measure == "mi":
            computed_global, computed_local = get_mi(destinations_data, sources_data)

            if trial_num == 0:
                summed_global = computed_global
                summed_local = computed_local
                for source in summed_local:
                    for destination in summed_local[source]:
                        summed_global[source][destination] = np.array(computed_global[source][destination])
                        summed_local[source][destination] = np.array(computed_local[source][destination])
            else:
                for source in summed_local:
                    for destination in summed_local[source]:
                        summed_global[source][destination] = summed_global[source][destination] + \
                                                             np.array(computed_global[source][destination])
                        summed_local[source][destination] = summed_local[source][destination] + \
                                                            np.array(computed_local[source][destination])

        elif measure == "ais":
            computed_global, computed_local, _ = get_ais(destinations_data)

            if trial_num == 0:
                summed_global = computed_global
                summed_local = computed_local
                for destination in summed_local:
                    summed_global[destination] = np.array(computed_global[destination])
                    summed_local[destination] = np.array(computed_local[destination])
            else:
                for destination in summed_local:
                    summed_global[destination] = summed_global[destination] + np.array(computed_global[destination])
                    summed_local[destination] = summed_local[destination] + np.array(computed_local[destination])

        elif measure == "cross_mi":
            left_motor = trial_data['keypress'][trial_num][:, 0].tolist()
            right_motor = trial_data['keypress'][trial_num][:, 1].tolist()

            brain_a1 = {'outputs_a1': outputs_a1, 'activation_a1': brain_states_a1}
            brain_a2 = {'outputs_a2': outputs_a2, 'activation_a2': brain_states_a2}

            computed_global21, computed_local21 = get_cross_mi(left_motor, brain_a2)
            computed_global12, computed_local12 = get_cross_mi(right_motor, brain_a1)

            computed_global = {'brain2_to_action1': computed_global21,
                               'brain1_to_action2': computed_global12}
            computed_local = {'brain2_to_action1': computed_local21,
                              'brain1_to_action2': computed_local12}

            if trial_num == 0:
                summed_global = computed_global
                summed_local = computed_local
                for predict_direction in summed_local:
                    for source in summed_local[predict_direction]:
                        summed_global[predict_direction][source] = np.array(computed_global[predict_direction][source])
                        summed_local[predict_direction][source] = np.array(computed_local[predict_direction][source])
            else:
                for predict_direction in summed_local:
                    for source in summed_local[predict_direction]:
                        summed_global[predict_direction][source] = summed_global[predict_direction][source] + \
                                        np.array(computed_global[predict_direction][source])
                        summed_local[predict_direction][source] = summed_local[predict_direction][source] + \
                                       np.array(computed_local[predict_direction][source])

    # get averages from sums

    avg_global = summed_global
    avg_local = summed_local

    if measure == "te" or measure == "mi":
        for source in summed_local:
            for destination in summed_local[source]:
                avg_global[source][destination] = summed_global[source][destination] / num_trials
                avg_local[source][destination] = summed_local[source][destination] / num_trials

    elif measure == "ais":
        for destination in summed_local:
            avg_global[destination] = summed_global[destination] / num_trials
            avg_local[destination] = summed_local[destination] / num_trials

    elif measure == "cross_mi":
        for predict_direction in summed_local:
            for source in summed_local[predict_direction]:
                avg_global[predict_direction][source] = summed_global[predict_direction][source] / num_trials
                avg_local[predict_direction][source] = summed_local[predict_direction][source] / num_trials

    return avg_global, avg_local


""" RUN """

# get data
pkl_file = open('resampled_td_914463.pkl', 'rb')
td = pickle.load(pkl_file)
pkl_file.close()

# pkl_file = open('td_immobile_914463.pkl', 'rb')
# td = pickle.load(pkl_file)
# pkl_file.close()


# jar location
jarLocation = "/Users/katja/javawd/infodynamics-dist-1.4/infodynamics.jar"
# start the JVM
jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)


averaged_te = get_avg_measures(td, "te")
output = open('avg_te_123.pkl', 'wb')
pickle.dump(averaged_te, output)
output.close()

# averaged_mi = get_avg_measures(resampled, "mi")
# output = open('avg_mi.pkl', 'wb')
# pickle.dump(averaged_mi, output)
# output.close()
#
# averaged_ais = get_avg_measures(resampled, "ais")
# output = open('avg_ais.pkl', 'wb')
# pickle.dump(averaged_ais, output)
# output.close()
#
# averaged_cross_mi = get_avg_measures(resampled, "cross_mi")
# output = open('avg_cross_mi.pkl', 'wb')
# pickle.dump(averaged_cross_mi, output)
# output.close()


jpype.shutdownJVM()
