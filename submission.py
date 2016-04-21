#!/usr/bin/python

import random
import collections
import math
import sys
from collections import Counter
from collections import defaultdict
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    wordList = x.split()
    return Counter(wordList)

############################################################
# Problem 3b: stochastic gradient descent

def gradientOfHingeLoss(features, y, w):
    '''
    Since we want the hinge loss we return 0 if 
    the margin is greater than 1(this means we are predicting correctly), 
    otherwise we return the gradient of the loss.
    '''
    margin = dotProduct(w,features)*y
    if margin >1 :
        return Counter()
    result = {}
    for key in features:
        result[key] = features[key] *-1*y
    return result



def learnPredictor(trainExamples, testExamples, featureExtractor):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, return the weight vector (sparse feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    numIters refers to a variable you need to declare. It is not passed in.
    '''
    numIters = 15
    stepSize = -.21
    lambdaShrink = .4
    weights = {}  # feature => weight
    featuresTrain = []
    featuresDev = []
    #for example in testExamples:
     #   entry = (featureExtractor(example[0]), example[1])
      #  featuresDev.append(entry)

    #for example in trainExamples:
     #   entry = (featureExtractor(example[0]), example[1])
      #  featuresTrain.append(entry)
   
    for i in range(numIters):
        for example in trainExamples:
            features= featureExtractor(example[0])
            gradient = gradientOfHingeLoss(features, example[1], weights)
            increment(weights,stepSize,gradient)
        
        def predictor(x):
            answer = dotProduct(weights, x)
            if answer >0 : return 1
            return -1
            
        #print "Iteration %d" %i
        #print "Training error = %f"  %evaluatePredictor(featuresTrain, predictor)
        #print "Dev error %f"% evaluatePredictor(featuresDev, predictor)
    return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)
    
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        #grab subset of keys in weights, give them a random value, multiply them by weights 
        limit = 5
        numFeatures = random.randint(1,len(weights))
        keys = random.sample(weights, numFeatures)
        phix = {}
        for key in keys:
            phix[key]= random.randint(1,limit)
        answer = dotProduct(weights, phix)
        if answer >0:  
            y = 1
        else : 
            y  = -1
        return (phix,y)
    result = []
    for i in range (numExamples): 
        result.append(generateExample())
    return result
   #

############################################################
# Problem 3f: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x):
        x = x.replace(" ","")
        ngram = x[:n]
        ngramFreq = Counter()
        ngramFreq[ngram] =1
        for letter in x[n:]:
            ngram = ngram[1:]
            ngram += letter
            ngramFreq[ngram] +=1
        return ngramFreq
    return extract
#print extractWordFeatures("hello there")
#Testing Gradient Descent
trainingExamples = readExamples("polarity.train")
testExamples = readExamples("polarity.dev")
learnPredictor(trainingExamples, testExamples, extractCharacterFeatures(6))
############################################################
# Problem 3h: extra credit features

def extractExtraCreditFeatures(x):
    # BEGIN_YOUR_CODE (around 1 line of code expected)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

############################################################
# Problem 4: k-means
############################################################

def vectorDifference(v1,v2):
    '''
    Returns the difference between two vectors.
    '''
    difference = defaultdict()
    for key in v1:
        if key in v2: difference[key] = v1[key] - v2[key]
        else: difference[key] = v1[key]
    for key in v2:
        if key not in v1: difference[key] = v2[key]   
    mySum =0
    for key in difference:
        mySum +=difference[key]**2
    return math.sqrt(mySum)
def incrementSparseVector(v1, scale, v2):
    """
    Given two sparse vectors |v1| and |v2|, perform v1 += scale * v2.
    This function will be useful later for linear classifiers.
    """
    result = {}
    for key2 in v2:
        if key2 in v1:
             v1[key2] = v1[key2] + scale*v2[key2]
        else:
            v1.setdefault(key2, scale*v2[key2]) 
def minClusterLoss(example, clusters):
    '''
    For a given example, and a list of clusters, this will find the index of the closest cluster,
    and will also find the error between the that cluster and the given exampel. Will return the pair (minIndex, minError)
    '''
    minLoss = sys.maxint
    minIndex =0
    for i, cluster in enumerate(clusters): #finding which cluster is closter to the example
        difference = vectorDifference(example, cluster)
        if difference < minLoss: 
            minLoss= difference
            minIndex = i
    return (minIndex, minLoss)

def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    assignments = [None]*len(examples)
    #assign relates the ith example to its current cluster
    random.seed(35)
    clusters = [None] * K

    ##PICK RANDOM INITIAL CLUSTERS
    for i in range (K):
        clusters[i] = examples[random.randint(0,len(examples)-1)]
    previousLoss = 0
    totalLoss = 0
    for x in range(maxIters):
        clusterMap =defaultdict(set)# maps cluster number to list of ints of indices of examples in cluster
        #STEP 1 Estimate Cluster Assignments
        totalLoss = 0
        for n in range(len(examples)):
            minIndex,minLoss = minClusterLoss(examples[n], clusters)
            assignments[n] = minIndex
            totalLoss+= minLoss**2
            clusterMap[minIndex].add(n)

    ## STEP 2
    ## Estimate Cluster Centers
        for n in range (len(clusters)):
            clusterSize = len(clusterMap[n])
            newCluster = {}
            # adding up all the words in the cluster
            for index in clusterMap[n]:
                incrementSparseVector(newCluster, 1, examples[index])
            for key in newCluster:
                newCluster[key] = newCluster[key]/float(clusterSize)
            clusters[n] = newCluster
        print totalLoss
        if totalLoss >= previousLoss and x is not 0: break ## it has converged
        previousLoss = totalLoss    
    return (clusters,assignments,totalLoss)

#Testing k means
numExamples = 10000
numWordsPerTopic = 3
numFillerWords = 1
examples = generateClusteringExamples(numExamples, numWordsPerTopic, numFillerWords)
(centroids,assignments,loss) = kmeans(examples,3,100)
print "total loss is"
print loss
#outputClusters("COOL_OUTPUT", examples, centroids, assignments)


