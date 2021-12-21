#!/usr/bin/python3
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
def loadData(path):
    colNames = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","PRICE",]
    dataSet = pd.read_csv(path,header=None,names=colNames,delim_whitespace=True,dtype=float,)
    return dataSet

def splitData(dataSet, testFrac):
    testData = dataSet.sample(frac=testFrac)
    trainData = dataSet.drop(testData.index)

    print("train:\t","cheap:\t",len(trainData[trainData.PRICE == 0]),"\texp:\t",len(trainData[trainData.PRICE == 1]),)
    print("test:\t","cheap:\t",len(testData[testData.PRICE == 0]),"\texp:\t",len(testData[testData.PRICE == 1]),)
    return trainData, testData

def preProcess(dataSet):
    meanPriceHouse = dataSet.PRICE.mean()
    dataSet.PRICE[dataSet.PRICE <= meanPriceHouse] = 0
    dataSet.PRICE[dataSet.PRICE > meanPriceHouse] = 1

    dataSet.loc[:, dataSet.columns != "PRICE"] = (
        dataSet.loc[:, dataSet.columns != "PRICE"]
        - dataSet.loc[:, dataSet.columns != "PRICE"].min()
    ) / (
        dataSet.loc[:, dataSet.columns != "PRICE"].max()
        - dataSet.loc[:, dataSet.columns != "PRICE"].min()
    )

    # dataSet.loc[:, dataSet.columns != "PRICE"] = (
    #     dataSet.loc[:, dataSet.columns != "PRICE"]
    #     - dataSet.loc[:, dataSet.columns != "PRICE"].mean()
    # ) / dataSet.loc[:, dataSet.columns != "PRICE"].std()
    return dataSet

def getProb(avg, std, x):
    return pd.DataFrame(
        1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.power((x - avg) / std, 2))).apply(np.prod,axis=1,)

def sigmoid(predictions):
    return 1 / (1 + np.exp(-predictions))

def pred(cheapProb, expensiveProb):
    df = pd.DataFrame(0, index=cheapProb.index, columns=["YHAT"], dtype=float)
    df[np.maximum(cheapProb, expensiveProb) == expensiveProb] = 1
    return df

def mle(trainData, testData):

    features = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT",]
    trainY = trainData.PRICE
    trainX = trainData[features]
    testY = testData.PRICE
    testX = testData[features]

    cheapProb = getProb(trainX[trainY == 0].mean(), trainX[trainY == 0].std(), testX,)

    expensiveProb = getProb(trainX[trainY == 1].mean(), trainX[trainY == 1].std(), testX,)
    # cheapProb = sigmoid(cheapProb)
    # expensiveProb = sigmoid(expensiveProb)

    predictions = pred(cheapProb, expensiveProb)
    print(predictions)
    accuracy = len(predictions[predictions.YHAT == testY]) / len(testY)
    cm = pd.DataFrame(0, index=["Actual No", "Actual Yes"], columns=["Predicted No", "Predicted Yes"], dtype=float,)

    cm["Predicted No"]["Actual No"] = len(
        predictions.loc[(predictions.YHAT == testY) & (predictions.YHAT == 0)])
    cm["Predicted Yes"]["Actual No"] = len(
        predictions.loc[(predictions.YHAT != testY) & (predictions.YHAT == 1)])
    cm["Predicted No"]["Actual Yes"] = len(
        predictions.loc[(predictions.YHAT != testY) & (predictions.YHAT == 0)])
    cm["Predicted Yes"]["Actual Yes"] = len(
        predictions.loc[(predictions.YHAT == testY) & (predictions.YHAT == 1)])
    return accuracy, cm

if __name__ == "__main__":
    dataSet = loadData("housing.csv")
    preProcesseddataSet = preProcess(dataSet)

    trainData, testData = splitData(preProcesseddataSet, testFrac=0.2)
    accuracy, cm = mle(trainData, testData)
    print("Accuracy:\t", accuracy, "\nCM:\n", cm)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    trainData, testData = splitData(preProcesseddataSet, testFrac=0.6)
    accuracy, cm = mle(trainData, testData)
    print("Accuracy:\t", accuracy, "\nCM:\n", cm)