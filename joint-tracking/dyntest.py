"""
subprocess.run(["ulimit", "-c", "unlimited"], shell=True)

Continuous data:
- binning misses the subtleties of data
- differential methods, 3 estimators:
-- Gaussian: only detects linear coupling; doesn't require parameters
-- box kernel: model-free (non-linearity included), sensitive to choosing radius r, biased
-- Kraskov KSG: nearest-neighbors, dynamic box width and bias correction; model-free
--- alg1: smaller var, larger bias
--- alg2: larger var, smaller bias

Usage:
- construct calculator
- set properties
- initialize
- set observations (setObservations OR addObservations between startadd and finaliseadd)
- compute
-- computeAverageLocalOfObservations - for one measure
-- computeLocalOfPreviousObservations - for local values (over time)
-- computeSignificance - for p-value

For KSG:
choose alg
set K (default is 4)

Significance testing:
how many samples? >= 3x # possible state configurations or >= the size of typical set
- test prob of sampling a given statistic I assuming X is independent of Y
-- form a surrogate distr Y with the same statistical properties but no relationship to X
-- measure p value
-- in practice: reshuffle Y (or bootstrap) many times to get a distr of I

calc.computeSignificance(num of surrogates)

Normalizing: not necessary if bias correction employed
Mi and TE for continuous data are theoretically independent of the scaling of individual variables,
and also the estimators work much better if the variables have the same scale.
So, basically always normalise.


Inf dynamics:
computation of the next state of a target variable in terms of information storage, transfer and modification
where does information of a rand var X_n+1 come from?

Measures:
1. entropy rate - how much inf form the past of X helps us predict its next step?
2. active information storage - how much inf about the next state of X can be found in its past state?
-> local AIS: inf from a specific past state in use in predicting specific next value
-> captures input driven dynamics

How to select embedding dimension (k history)?
- minimizing prediction error
- maximizing AIS (for KSG estimator bias correction is built in)
- non-uniform embedding

KSG calculator provides a property to find embedding automatically using one of these options

3. predictive inf - how much inf about the future X_n+1(k+) can be found in its past state
(at some point rather than at next state)
4. excess entropy - total memory


AIS is about information storage, accounts for the past of X that helps us predict its next state (of X)
Inf transfer is about predicting the next state of X based on the past of Y (source) in the context of past state of X
--> transfer entropy

TE vs (lagged) MI?
- TE is conditioned on the past of X
- TE is dynamic


Steps:
1. set target embedding
- embed target first, not target and source together (can choose value in property)
2. set source embedding
3. set delay


Embedding can be also selected manually:
- investigate AIS as a function of embedded history length k
- for a given number of K nearest neighbours (here 4), plot AIS (calculated with Kraskov MI) as a function of k

Since the Kraskov estimator is bias corrected (i.e. while a raw MI would normally increase with an increase in
dimensionality -- caused by increasing k here -- this one will reduce once we start undersampling), we can use the
peak to suggest that an embedded history of k=2 for both heart and breath time-series is appropriate to capture all
relevant information from the past without capturing more spurious than relevant information as k increases. This
result is stable with the number of nearest neighbours K -- you can check this by changing K=4 above.

For our given embedding lengths, we compute TE (using KSG) for a number of K nearest neighbours and plot it


Auto-embedding for multi-variate measures:
- not implemented for TE
- only Ragwitz for AIS
One could use multivariate AIS to embed each of the source and destination (separately) the way univariate TE does.


JIDT on the other hand offers two options:
1. Embed the source and destination independently, using the Ragwitz criteria applied separately to each one.
TransferEntropyCalculatorKraskov.PROP_AUTO_EMBED_METHOD =
TransferEntropyCalculatorKraskov.AUTO_EMBED_METHOD_RAGWITZ

2. Embed the destination alone using the Ragwitz critera.
TransferEntropyCalculatorKraskov.PROP_AUTO_EMBED_METHOD =
TransferEntropyCalculatorKraskov.AUTO_EMBED_METHOD_RAGWITZ_DEST_ONLY

Another option is to use AIS criterion:
- Use an AIS Kraskov calculator to auto-embed the destination, retrieving the auto-embedded
parameters using getProperty() after setObservations().
- Construct the TransferEntropyCalculatorKraskov, not use auto-embedding, but directly supply the
parameters fitted for the destination using the AIS calculator.


Should source be embedded or should we use single source values (i.e. setting l=1)?
Depends on the situation with your data:
A. If you have direct access to the values of your variables (i.e. not noise / measurement obscured etc, but probably
obtained from simulation), and you know that your source and target are directly causally linked, and you know that
this direct causal link only exists across one time delay, then source should not be embedded.
This makes sure you are only measuring information directly transferred at each step.
B. On the other hand -- if you only have access to some noisy projections of your variables, or they represent some
higher dimensional state of the underlying source system (e.g. most neuroscience data sets), or you are unsure of the
nature of the causal link (direct or indirect, lag or multiple lags), then embedding the source is definitely the way
to go. This is because the embedding (if done properly) builds you a vector representing the underlying state of the
source (a Takens embedding vector -- see the description in the paper introducing JIDT). This is a better representation
of what actually (causally) interacts with the destination, and this is generally what you want to capture in the TE.


create teCalc object
set any required properties # including the auto embedding parameters
initialise storage for embedding length ("k") and delay ("k_tau") for each variable, and track whether they have been
determined yet for each variable
For each column (source?):  (Notice I've swapped these loops to minimise our use of auto-embedding)

    For each row (destination?):
        if embedding parameters not determined yet for the destination variable, set the property for autoembedding to
        AUTO_EMBED_METHOD_RAGWITZ_DEST_ONLY
               if embedding parameters not determined yet for the source variable as well, set the property for
               autoembedding to AUTO_EMBED_METHOD_RAGWITZ
        Else, turn off auto embedding, retrieve the auto embedding parameters set previously for these variables and
        set them either via setProperties or via the initialise() method
        Initialize()  # You need to do this before every new calculation

        teCalc.setObservations

        teCalc.computeAverageLocalOfObservations()
        Ask the teCalc what the autoembedding properties were set to (you can do this via calls to the getProperty()
        method, e.g. getProperty("k_HISTORY"), and store these for the relevant variables.


If a target is completely predictable from its own past, TE goes to zero. In such a case, MI is a better measure.

Conditional TE
- adding var Z in conditioning X on Y
- common driver, pathway and synergy effects


For network inference use IDTxl package
- examine how network changes over time
- uses TE
- measure pairwise te between all vars
- select those above some threshold (standard: threshold pvalue, with bonferroni)
- better to use iterative approach, included in the idtxl


for different lags - calculate in a loop (property TIME_DIFF);
for negative lag, invert source and destination


Basic use
from jpype import *
startJVM(getDefaultJVMPath(), "-Djava.class.path=" + '/Users/katja/javawd/infodynamics-dist-1.4/infodynamics.jar')
teCalcClass = JPackage("infodynamics.measures.discrete").TransferEntropyCalculatorDiscrete
teCalc = teCalcClass(2,1)
# add observations
teCalc.initialise()
# compute
shutdownJVM()
"""
import jpype
import numpy as np
from network_analyzer.readFloatsFile import readFloatsFile

# jar location
jarLocation = "/Users/katja/javawd/infodynamics-dist-1.4/infodynamics.jar"
# Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)


"""Discrete Active Information Storage"""

# # 0. Load/prepare the data
# dataRaw = readIntsFile.readIntsFile("/Users/katja/javawd/infodynamics-dist-1.4/demos/data/2coupledBinaryColsUseK2.txt")
# # As numpy array:
# data = np.array(dataRaw)
# variable = jpype.JArray(jpype.JInt, 1)(data[:,0].tolist())
#
# # 1. Construct the calculator:
# calcClass = jpype.JPackage("infodynamics.measures.discrete").ActiveInformationCalculatorDiscrete
# calc = calcClass(2, 2)
# # 2. No other properties to set for discrete calculators.
# # 3. Initialise the calculator for (re-)use:
# calc.initialise()
# # 4. Supply the sample data:
# calc.addObservations(variable)
# # 5. Compute the estimate:
# result = calc.computeAverageLocalOfObservations()
# # 6. Compute the (statistical significance via) null distribution empirically (e.g. with 100 permutations):
# measDist = calc.computeSignificance(100)
#
# print("AIS_Discrete(col_0) = %.4f bits (null: %.4f +/- %.4f std dev.; p(surrogate > measured)=%.5f from %d surrogates)" %
#     (result, measDist.getMeanOfDistribution(), measDist.getStdOfDistribution(), measDist.pValue, 100))


"""Continuous Active Information Storage (Kraskov)"""

# # 0. Load/prepare the data:
# dataRaw = readFloatsFile.readFloatsFile("/Users/katja/javawd/infodynamics-dist-1.4/demos"
#                                         "/data/SFI-heartRate_breathVol_bloodOx.txt")
# # As numpy array:
# data = np.array(dataRaw)
# variable = data[:, 0]
#
# # 1. Construct the calculator:
# calcClass = jpype.JPackage("infodynamics.measures.continuous.kraskov").ActiveInfoStorageCalculatorKraskov
# calc = calcClass()
# # 2. Set any properties to non-default values:
# calc.setProperty("AUTO_EMBED_METHOD", "MAX_CORR_AIS")
# calc.setProperty("AUTO_EMBED_K_SEARCH_MAX", "6")
# calc.setProperty("AUTO_EMBED_TAU_SEARCH_MAX", "6")
# # 3. Initialise the calculator for (re-)use:
# calc.initialise()
# # 4. Supply the sample data:
# calc.setObservations(variable)
# # 5. Compute the estimate:
# result = calc.computeAverageLocalOfObservations()
#
# print("AIS_Kraskov (KSG)(col_0) = %.4f nats" % result)


"""Discrete Transfer Entropy"""

# # Generate some random binary data.
# sourceArray = [random.randint(0,1) for r1 in range(100)]
# destArray = [0] + sourceArray[0:99]
# sourceArray2 = [random.randint(0,1) for r2 in range(100)]
#
# # Create a TE calculator and run it:
# teCalcClass = jpype.JPackage("infodynamics.measures.discrete").TransferEntropyCalculatorDiscrete
# teCalc = teCalcClass(2,1)
# teCalc.initialise()
#
# # First use simple arrays of ints, which we can directly pass in:
# teCalc.addObservations(sourceArray, destArray)
# print("For copied source, result should be close to 1 bit : %.4f" % teCalc.computeAverageLocalOfObservations())
# teCalc.initialise()
# teCalc.addObservations(sourceArray2, destArray)
# print("For random source, result should be close to 0 bits: %.4f" % teCalc.computeAverageLocalOfObservations())
#
# # Next, demonstrate how to do this with a numpy array
# teCalc.initialise()
# # Create the numpy arrays:
# sourceNumpy = np.array(sourceArray, dtype=np.int)
# destNumpy = np.array(destArray, dtype=np.int)
# sourceNumpyJArray = jpype.JArray(jpype.JInt, 1)(sourceNumpy.tolist())
# destNumpyJArray = jpype.JArray(jpype.JInt, 1)(destNumpy.tolist())
# teCalc.addObservations(sourceNumpyJArray, destNumpyJArray)
# print("Using numpy array for copied source, result confirmed as: %.4f" % teCalc.computeAverageLocalOfObservations())
#
# # calculate significance
# EmpiricalMeasurementDistribution = teCalc.computeSignificance(1000)
# print(EmpiricalMeasurementDistribution)
#
# # calculate and plot local TE over time
# l1 = teCalc.computeLocal(sourceNumpyJArray, destNumpyJArray)
# plt.plot(list(l1))
# plt.show()


"""Continuous Transfer Entropy (Kraskov)"""

# # Generate some random normalised data.
# numObservations = 1000
# covariance = 0.4
# # Source array of random normals:
# sourceArray = [random.normalvariate(0,1) for r in range(numObservations)]
# # Destination array of random normals with partial correlation to previous value of sourceArray
# destArray = [0] + [sum(pair) for pair in zip([covariance*y for y in sourceArray[0:numObservations-1]],
#                                              [(1-covariance)*y for y in [random.normalvariate(0,1)
#                                                                          for r3 in range(numObservations-1)]])]
# # Uncorrelated source array:
# sourceArray2 = [random.normalvariate(0,1) for r4 in range(numObservations)]
#
# # Create a TE calculator and run it:
# teCalcClass = jpype.JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
# teCalc = teCalcClass()
# teCalc.setProperty("NORMALISE", "true")  # Normalise the individual variables
# teCalc.setProperty("AUTO_EMBED_METHOD", "MAX_CORR_AIS")
# teCalc.setProperty("AUTO_EMBED_K_SEARCH_MAX", "10")
# teCalc.setProperty("k", "4")  # Use Kraskov parameter K=4 for 4 nearest points
#
# teCalc.initialise(1)  # Use history length 1 (Schreiber k=1)
#
# # Perform calculation with correlated source:
# teCalc.setObservations(jpype.JArray(jpype.JDouble, 1)(sourceArray), jpype.JArray(jpype.JDouble, 1)(destArray))
# result = teCalc.computeAverageLocalOfObservations()
# # Note that the calculation is a random variable (because the generated data is a set of random variables) -
# # the result will be of the order of what we expect, but not exactly equal to it; in fact, there will
# # be a large variance around it.
# expected_result = math.log(1/(1-math.pow(covariance, 2)))
#
# print("TE result {} nats; expected to be close to {} nats for these correlated Gaussians".format(result,
#                                                                                                  expected_result))
#
# # Perform calculation with uncorrelated source:
# teCalc.initialise()  # Initialise leaving the parameters the same
# teCalc.setObservations(jpype.JArray(jpype.JDouble, 1)(sourceArray2), jpype.JArray(jpype.JDouble, 1)(destArray))
# result2 = teCalc.computeAverageLocalOfObservations()
# print("TE result %.4f nats; expected to be close to 0 nats for these uncorrelated Gaussians" % result2)


"""Continuous Transfer Entropy with automatic embedding (Kraskov)"""

# Examine the heart-breath interaction that Schreiber originally looked at:
datafile = "/Users/katja/javawd/infodynamics-dist-1.4/demos/data/SFI-heartRate_breathVol_bloodOx.txt"
data = readFloatsFile.readFloatsFile(datafile)
# As numpy array:
A = np.array(data)
# Select data points 2350:3550, pulling out the relevant columns:
breathRate = A[2350:3551, 1]
heartRate = A[2350:3551, 0]

# Create a Kraskov TE calculator:
teCalcClass = jpype.JPackage("infodynamics.measures.continuous.kraskov").TransferEntropyCalculatorKraskov
teCalc = teCalcClass()

# Set properties for auto-embedding of both source and destination
#  using the Ragwitz criteria:
#  a. Auto-embedding method
teCalc.setProperty(teCalcClass.PROP_AUTO_EMBED_METHOD, teCalcClass.AUTO_EMBED_METHOD_RAGWITZ)
#  b. Search range for embedding dimension (k) and delay (tau)
teCalc.setProperty(teCalcClass.PROP_K_SEARCH_MAX, "6")
teCalc.setProperty(teCalcClass.PROP_TAU_SEARCH_MAX, "6")
# Since we're auto-embedding, no need to supply k, l, k_tau, l_tau here:
teCalc.initialise()
# Compute TE from breath (column 1) to heart (column 0)
teCalc.setObservations(breathRate, heartRate)
teBreathToHeart = teCalc.computeAverageLocalOfObservations()

# Check the auto-selected parameters and print out the result:
optimisedK = int(teCalc.getProperty(teCalcClass.K_PROP_NAME))
optimisedKTau = int(teCalc.getProperty(teCalcClass.K_TAU_PROP_NAME))
optimisedL = int(teCalc.getProperty(teCalcClass.L_PROP_NAME))
optimisedLTau = int(teCalc.getProperty(teCalcClass.L_TAU_PROP_NAME))
print(("TE(breath->heart) was %.3f nats for (heart embedding:) k=%d," +
       "k_tau=%d, (breath embedding:) l=%d,l_tau=%d optimised via Ragwitz criteria") %
      (teBreathToHeart, optimisedK, optimisedKTau, optimisedL, optimisedLTau))

# Next, embed the destination only using the Ragwitz criteria:
teCalc.setProperty(teCalcClass.PROP_AUTO_EMBED_METHOD, teCalcClass.AUTO_EMBED_METHOD_RAGWITZ_DEST_ONLY)
teCalc.setProperty(teCalcClass.PROP_K_SEARCH_MAX, "6")
teCalc.setProperty(teCalcClass.PROP_TAU_SEARCH_MAX, "6")
# Since we're only auto-embedding the destination, we supply
#  source embedding here (to overwrite the auto embeddings from above):
teCalc.setProperty(teCalcClass.L_PROP_NAME, "1")
teCalc.setProperty(teCalcClass.L_TAU_PROP_NAME, "1")
# Since we're auto-embedding, no need to supply k and k_tau here:
teCalc.initialise()
# Compute TE from breath (column 1) to heart (column 0)
teCalc.setObservations(breathRate, heartRate)
teBreathToHeartDestEmbedding = teCalc.computeAverageLocalOfObservations()

# Check the auto-selected parameters and print out the result:
optimisedK = int(teCalc.getProperty(teCalcClass.K_PROP_NAME))
optimisedKTau = int(teCalc.getProperty(teCalcClass.K_TAU_PROP_NAME))
print(("TE(breath->heart) was %.3f nats for (heart embedding:) k=%d," +
       "k_tau=%d, optimised via Ragwitz criteria, plus (breath embedding:) l=1,l_tau=1") %
      (teBreathToHeartDestEmbedding, optimisedK, optimisedKTau))

jpype.shutdownJVM()
