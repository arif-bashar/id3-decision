"""
author: Arif Bashar

Create an ID3 decision tree to classify some datasets
"""

import sys
import numpy as np

def sortColumn(data, column):
    return data[data[:,column].argsort()]

# Return a list of probabilities for each class in the last column
def getClassProb(data):
    classes = np.unique(data[:, -1])
    probabilities = []
    for index in range(len(classes)):
        count = np.count_nonzero(data[:,-1] == classes[index])
        probabilities.append(count/len(data))
    return probabilities

# Get information of our class labels
def getInfo(data):
    probabilities = getClassProb(data)

    info = sum(probabilities * -np.log2(probabilities))
    return info

# Determining potential binary split points based on attribute value changes
# Return dictionary containing potential split points
def getPotSplits(data):

    """
    Using a dictionary to store the potential splits for each column
    key: column  --  value: split point
    
    """

    splits = {}
    _ , colSize = data.shape
    colSize -= 1

    for colIndex in range(colSize):
        data = sortColumn(data, colIndex)
        splits[colIndex] = []   # Use a list to store all potential splits 
        values = data[:, colIndex]  # Grab all values for the column
        uniqueValues = np.unique(values)    # Get rid of duplicates (we only want value changes)

        # Enter this loop to calculate actual split points
        for index in range(len(uniqueValues)):
            if index > 0:   # Skip index 0 because we can't get its previous 
                current = uniqueValues[index] 
                previous = uniqueValues[index-1]
                split = (current + previous) / 2
                splits[colIndex].append(split)

    return splits

# Average the two values to make the split: those examples less than the split 
# value and those examples greater than or equal to the split value
# Return two lists: all values above split and all values below specified split
def splitData(data, column, splitValue):
    values = data[:, column]
    greaterValues = data[values >= splitValue] 
    lesserValues = data[values < splitValue]

    return greaterValues, lesserValues

# Calculate the entropy so we can determine the best split
def getEntropy(greaterData, lesserData):
    dataCount = len(greaterData) + len(lesserData)
    greaterProb = len(greaterData) / dataCount
    lesserProb = len(lesserData) / dataCount
    entropy = (lesserProb * getInfo(lesserData) + greaterProb * getInfo(greaterData))

    return entropy

# Determine the best split where 
def getBestSplit(data, potentialSplits):
    # We will decide the best split based on max information gain
    maxGain = 0

    for column in potentialSplits:
        data = sortColumn(data, column)
        for split in potentialSplits[column]:
            greaterValues, lesserValues = splitData(data, column, split)
            entropy = getEntropy(greaterValues, lesserValues)
            gain = getInfo(data) - entropy

            if (maxGain < gain):
                maxGain = gain
                bestSplit = split
                bestColumn = column

    return bestColumn, bestSplit, maxGain

# Check terminal cases
def isTerminal(data):
    labelColumn = data[:, -1]
    classes = np.unique(labelColumn)

    if len(classes) == 1:
        return True
    else:
        return False

# Classify the data given what is in the last column
def classify(data):
    classes = np.unique(data[:, -1])
    uniqueClasses, uniqueCount = np.unique(classes, return_counts=True)
    
    index = uniqueCount.argmax()
    classification = uniqueClasses[index]

    return classification

# Main recursive algorithm to build our decision tree
def buildTree(data):
    if isTerminal(data):
        return classify(data)
        
    else:
        potentialSplits = getPotSplits(data)
        bestColumn, bestSplit, _ = getBestSplit(data, potentialSplits)
        greaterValues, lesserValues = splitData(data, bestColumn, bestSplit)

        question = "{} <= {}".format(bestColumn, bestSplit)
        tree = {question: []}
        leftNode = buildTree(lesserValues)
        rightNode = buildTree(greaterValues)

        if rightNode == leftNode:
            tree = rightNode
        else:
            tree[question].append(leftNode)
            tree[question].append(rightNode)

    return tree

# Use testing data and try to predict its classification one row at a time
def predict(testData, tree):
    question = list(tree.keys())[0]
    col, comparison, value = question.split(" ")

    # Ask the question
    if comparison == "<=":
        if testData[int(col)] <= float(value):
            prediction = tree[question][0]
        else:
            prediction = tree[question][1]

    # Base case for recursion
    if not isinstance(prediction, dict):
        return prediction
    
    # Recurse through
    else:
        remainingTree = prediction
        return predict(testData, remainingTree)

# Returns number of correct predictions
def getAccuracy(data, tree):
    accuracy = 0
    for row in range(len(data)):
        prediction = predict(data[row], tree)
        if prediction == data[row][-1]:
            accuracy += 1
    return accuracy
        
def main():
    trainDataName = (sys.argv[1])
    testDataName = (sys.argv[2])

    train = np.loadtxt(trainDataName)
    test = np.loadtxt(testDataName)

    if len(train.shape) < 2:
        train = np.array([train])
    if len(test.shape) < 2:
        test = np.array([test])

    tree = buildTree(train)
    print(tree)
    # print(getAccuracy(test, tree))

    

main()