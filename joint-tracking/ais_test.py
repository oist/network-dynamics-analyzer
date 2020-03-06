import jpype
import matplotlib.pyplot as plt
import pickle


# jar location
jarLocation = "/Users/katja/javawd/infodynamics-dist-1.4/infodynamics.jar"
# Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

# 0. Load/prepare the data:
pkl_file = open('resampled_td_914463.pkl', 'rb')
td = pickle.load(pkl_file)
pkl_file.close()

outputs_a1_n8 = td['output_a1'][2][:, 7].tolist()
brain_states_a8 = td['brain_state_a1'][2][:, 7].tolist()


# 1. Construct the calculator:
ais_calc_class = jpype.JPackage("infodynamics.measures.continuous.kraskov").ActiveInfoStorageCalculatorKraskov
ais_calc = ais_calc_class()

# 2. Set any properties to non-default values:
ais_calc.setProperty("AUTO_EMBED_METHOD", "MAX_CORR_AIS")
ais_calc.setProperty("AUTO_EMBED_K_SEARCH_MAX", "6")
ais_calc.setProperty("AUTO_EMBED_TAU_SEARCH_MAX", "6")


# 3. Initialise the calculator for (re-)use:
ais_calc.initialise()
# 4. Supply the sample data:
ais_calc.setObservations(outputs_a1_n8)

# 5. Compute the estimate:
ais = ais_calc.computeAverageLocalOfObservations()
output_local_measures = ais_calc.computeLocalOfPreviousObservations()
optimised_k = int(ais_calc.getProperty(ais_calc_class.K_PROP_NAME))
optimised_k_tau = int(ais_calc.getProperty(ais_calc_class.TAU_PROP_NAME))
print(optimised_k)
print(optimised_k_tau)
print(ais)


# 3. Initialise the calculator for (re-)use:
ais_calc.initialise()
ais_calc.setObservations(brain_states_a8)
# 5. Compute the estimate:
ais = ais_calc.computeAverageLocalOfObservations()
activation_local_measures = ais_calc.computeLocalOfPreviousObservations()
optimised_k = int(ais_calc.getProperty(ais_calc_class.K_PROP_NAME))
optimised_k_tau = int(ais_calc.getProperty(ais_calc_class.TAU_PROP_NAME))
print(optimised_k)
print(optimised_k_tau)
print(ais)

plt.figure()
plt.plot(output_local_measures, label="output")
plt.plot(activation_local_measures, label="activation")
plt.legend()
plt.savefig('ais_test.png')

jpype.shutdownJVM()

