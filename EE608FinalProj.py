# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 21:38:59 2018

@author: Sourav Kumar Ukil
"""

'''
this document isn't really meant to be run completely (at once).
some variables are reused (copied code that was used again) which
can overwrite earlier results. If run, it should be done in chunks. 
'''
from mpl_toolkits.mplot3d import Axes3D
import sklearn
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.RandomState(42)
import math

# read data
data = pd.read_csv('608_Data.csv', header = 0, encoding='latin1')

# cutting off the country collectives (Arab World, etc.)
countriesData = data[25:]

# idea: fill missing values in with the average of all the countries

# there are too many to drop the missing entries (at least without reducing
# our sample size further)

# fills in the matrix columns with the average of the column
def missingValueFiller(vector):
    result = []
    # sum and count are used to find the average value
    summation = 0
    count = 0
    for i in range(0, len(vector)):
        if math.isnan(vector[i]) == False:
            summation = summation + vector[i]
            count += 1
    averageValue = summation / count
    # if the vector is nan then use the average otherwise use hte original value
    for i in range(0, len(vector)):
        if math.isnan(vector[i]):
            result.append(averageValue)
        else:
            result.append(vector[i])
    return result       



### following chunk isn't used
'''
filledinData = np.array(countriesData)
for i in range(1, 14):
    filledinData[:,i] = missingValueFiller(np.array(countriesData)[:,i])
 now we have all the missing values filled in! 
 graphing it to see what we have
 plt.plot(filledinData[:,5], filledinData[:,1], 'ro')
'''


# Find all entries that include a response
countnaninresponse = 0
response = []
indices = []
for i in range(0, 217):
    if (math.isnan(np.array(countriesData)[i,1])):
        countnaninresponse += 1
    else:
        response.append(np.array(countriesData)[i,1])
        indices.append(i)

# Use missing value filler and the entries found above to create the dataset
# we're going to use to train and evaluate the model
validCountries = list(np.array(countriesData)[indices,:])
filledinValid = np.array(countriesData)[indices,:]
for i in range(2, 15):
    filledinValid[:,i] = missingValueFiller(np.array(countriesData)[indices,i])

# create a new features youngPopRatio which finds the ratio of young & senior 
# persons (<15, >64) compared to the total population 
def youngPopulationRatio(adultPop, totalPop):
    youngPopRatio = []
    for i in range(0, len(totalPop)):
        if (1 - (adultPop[i]) / totalPop[i]) > 0:
            youngPopRatio.append(1 - (adultPop[i] / totalPop[i]))
        else:
            youngPopRatio.append(.3546) # sometimes this gives wildly negative numbers (-1500) so replacing
                                  # those with a number close to the mean
    return youngPopRatio

youngPopRatioFemale = missingValueFiller(np.array(youngPopulationRatio(filledinValid[:,8], filledinValid[:,9])))
youngPopRatioMale = missingValueFiller(np.array(youngPopulationRatio(filledinValid[:,10], filledinValid[:,11])))
youngPopRatio = missingValueFiller(np.array(youngPopulationRatio(filledinValid[:,12], filledinValid[:,13])))
newFilledinValid = np.hstack((filledinValid,np.zeros([130,3])))
newFilledinValid[:,14] = youngPopRatioFemale
newFilledinValid[:,15] = youngPopRatioMale
newFilledinValid[:,16] = youngPopRatio

# Some plots to evaluate the relationships between the variables

plt.figure()
plt.plot(countriesData[:,5], filledinValid[:,1], 'ro')
plt.ylabel('Net Enrollment Rate')
plt.xlabel('GDP per capita)')

plt.figure()
plt.plot(filledinValid[:,6], filledinValid[:,1], 'ro')
plt.ylabel('Net Enrollment Rate')
plt.xlabel('GNI per capita')

plt.figure()
plt.plot(filledinValid[:,7], filledinValid[:,1], 'ro')
plt.ylabel('Net Enrollment Rate')
plt.xlabel('Internet Users per 100 persons')

plt.figure()
plt.plot(youngPopRatio, filledinValid[:,1], 'ro')
plt.ylabel('Net Enrollment Rate')
plt.xlabel('Young and Old Population Ratio (Population < 15 or > 64 / Population)')

### Population graphs
plt.figure()
plt.plot(filledinValid[:,8], filledinValid[:,1], 'ro')
plt.ylabel('Net Enrollment Rate')
plt.xlabel('Population Female 15-64')

plt.figure()
plt.plot(filledinValid[:,9], filledinValid[:,1], 'ro')
plt.ylabel('Net Enrollment Rate')
plt.xlabel('Population Female')

plt.figure()
plt.plot(filledinValid[:,10], filledinValid[:,1], 'ro')
plt.ylabel('Net Enrollment Rate')
plt.xlabel('Population Male 15-64')

plt.figure()
plt.plot(filledinValid[:,11], filledinValid[:,1], 'ro')
plt.ylabel('Net Enrollment Rate')
plt.xlabel('Population Male')

plt.figure()
plt.plot(filledinValid[:,12], filledinValid[:,1], 'ro')
plt.ylabel('Net Enrollment Rate')
plt.xlabel('Population Total 15-64')

plt.figure()
plt.plot(filledinValid[:,13], filledinValid[:,1], 'ro')
plt.ylabel('Net Enrollment Rate')
plt.xlabel('Total Population')


'''
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(youngPopRatio, filledinValid[:,5], filledinValid[:,1])
'''     



## splitting into training vs test

# Random indices for training vs. test
train = rng.randint(0, 130, 130)[0:100,np.newaxis]
test = rng.randint(0, 130, 130)[100:,np.newaxis]

# Response for netenrollment predictions
netEnrollmentResponseTrain = newFilledinValid[train,1]
netEnrollmentResponseTest = newFilledinValid[test,1]

# Response for net enrollment male - female
netEnrollmentSResponseTrain = newFilledinValid[train,2] - newFilledinValid[train,3]
netEnrollmentSResponseTest = newFilledinValid[test,2] - newFilledinValid[test,3]



## Modeling using 4 features

# Features GDP per capite (5), GNI per capita (6), Internet Users (7), youngPopRatio (16) 
features = [5, 6, 7, 16]
netEnrollmentFeaturesTrain = newFilledinValid[train,features]
netEnrollmentFeaturesTest = newFilledinValid[test,features]

# Linear regression training
regression = linear_model.LinearRegression()
regression.fit(netEnrollmentFeaturesTrain, netEnrollmentResponseTrain)

# Make predictions
prediction = regression.predict(netEnrollmentFeaturesTest)
predictionTrain = regression.predict(netEnrollmentFeaturesTrain)

coefficients = regression.coef_
meansquarederror = sklearn.metrics.mean_squared_error(netEnrollmentResponseTest, prediction)
rTest = sklearn.metrics.r2_score(netEnrollmentResponseTest, prediction)
rTrain = sklearn.metrics.r2_score(netEnrollmentResponseTrain, predictionTrain)

# Try standardizing features first

scaler = StandardScaler()
scalerTrain = list(netEnrollmentFeaturesTrain)
scalerTest = list(netEnrollmentFeaturesTest)
scaler.fit(scalerTrain)
netEnrollmentFeaturesTrain2 = scaler.transform(scalerTrain)
netEnrollmentFeaturesTest2 = scaler.transform(scalerTest)

# Linear regression training
regression2 = linear_model.LinearRegression()
regression2.fit(netEnrollmentFeaturesTrain2, netEnrollmentResponseTrain)

# Make predictions
prediction2Test = regression2.predict(netEnrollmentFeaturesTest2)
prediction2Train = regression2.predict(netEnrollmentFeaturesTrain2)

coefficients2 = regression2.coef_
meansquarederror2 = sklearn.metrics.mean_squared_error(netEnrollmentResponseTest, prediction2Test)
r2Test = sklearn.metrics.r2_score(netEnrollmentResponseTest, prediction2Test)
r2Train = sklearn.metrics.r2_score(netEnrollmentResponseTrain, prediction2Train)

# standardizing doesn't affect results, which makes sense

# Linear regression doesn't seem to be working well, which was unlikely considering
# the graphs. A curve would work better, maybe we can fit some polynomial

polyRegression = sklearn.preprocessing.PolynomialFeatures(2)
netEnrollmentPolyFeaturesTrain = polyRegression.fit_transform(netEnrollmentFeaturesTrain)
netEnrollmentPolyFeaturesTest = polyRegression.fit_transform(netEnrollmentFeaturesTest)

# Linear regression training
regressionPoly = linear_model.LinearRegression()
regressionPoly.fit(netEnrollmentPolyFeaturesTrain, netEnrollmentResponseTrain)

# Make predictions
predictionPolyTest = regressionPoly.predict(netEnrollmentPolyFeaturesTest)
predictionPolyTrain = regressionPoly.predict(netEnrollmentPolyFeaturesTrain)

coefficientsPoly = regressionPoly.coef_
meansquarederrorPoly = sklearn.metrics.mean_squared_error(netEnrollmentResponseTest, predictionPolyTest)
rPolyTest = sklearn.metrics.r2_score(netEnrollmentResponseTest, predictionPolyTest)
rPolyTrain = sklearn.metrics.r2_score(netEnrollmentResponseTrain, predictionPolyTrain)
# best so far

# log transformation before regression

logTransformer = FunctionTransformer(np.log1p)
netEnrollmentLogFeaturesTrain = logTransformer.transform(netEnrollmentFeaturesTrain)
netEnrollmentLogFeaturesTest = logTransformer.transform(netEnrollmentFeaturesTest)

# regression training
regressionLog = linear_model.LinearRegression()
regressionLog.fit(netEnrollmentFeaturesTrain, netEnrollmentResponseTrain)

# predictions
predictionLogTest = regressionLog.predict(netEnrollmentLogFeaturesTest)
predictionLogTrain = regressionLog.predict(netEnrollmentLogFeaturesTrain)

rLogTest = sklearn.metrics.r2_score(netEnrollmentResponseTest, predictionLogTest)
rLogTrain = sklearn.metrics.r2_score(netEnrollmentResponseTrain, predictionLogTrain)
# both negative, so quite bad

############
############

## not used
'''
polyRegression = sklearn.preprocessing.PolynomialFeatures(2)
netEnrollmentPolyFeaturesTrain = polyRegression.fit_transform(netEnrollmentFeaturesTrain2)
netEnrollmentPolyFeaturesTest = polyRegression.fit_transform(netEnrollmentFeaturesTest2)

# Linear regression training
regressionPoly = linear_model.LinearRegression()
regressionPoly.fit(netEnrollmentPolyFeaturesTrain, netEnrollmentResponseTrain)

# Make predictions
predictionPolyTest = regressionPoly.predict(netEnrollmentPolyFeaturesTest)
predictionPolyTrain = regressionPoly.predict(netEnrollmentPolyFeaturesTrain)

coefficientsPoly = regressionPoly.coef_
meansquarederrorPoly = sklearn.metrics.mean_squared_error(netEnrollmentResponseTest, predictionPolyTest)
rPolyTest = sklearn.metrics.r2_score(netEnrollmentResponseTest, predictionPolyTest)
rPolyTrain = sklearn.metrics.r2_score(netEnrollmentResponseTrain, predictionPolyTrain)
'''

# Trying with more features
moreFeatures = [5, 6, 7, 8, 9, 10, 11, 12, 13, 16]
netEnrollmentMoreFeaturesTrain = newFilledinValid[train, moreFeatures]
netEnrollmentMoreFeaturesTest = newFilledinValid[test, moreFeatures]

# Linear regression training
regression = linear_model.LinearRegression()
regression.fit(netEnrollmentMoreFeaturesTrain, netEnrollmentResponseTrain)

# Make predictions
prediction = regression.predict(netEnrollmentMoreFeaturesTest)
predictionTrain = regression.predict(netEnrollmentMoreFeaturesTrain)

coefficients = regression.coef_
meansquarederror = sklearn.metrics.mean_squared_error(netEnrollmentResponseTest, prediction)
rTest2 = sklearn.metrics.r2_score(netEnrollmentResponseTest, prediction)
rTrain2 = sklearn.metrics.r2_score(netEnrollmentResponseTrain, predictionTrain)

# A bit better than previous stuff. Now try polynomial version

polyRegression = sklearn.preprocessing.PolynomialFeatures(2)
netEnrollmentPolyFeaturesTrain = polyRegression.fit_transform(netEnrollmentMoreFeaturesTrain)
netEnrollmentPolyFeaturesTest = polyRegression.fit_transform(netEnrollmentMoreFeaturesTest)

# Linear regression training
regressionPoly = linear_model.LinearRegression()
regressionPoly.fit(netEnrollmentPolyFeaturesTrain, netEnrollmentResponseTrain)

# Make predictions
predictionPolyTest = regressionPoly.predict(netEnrollmentPolyFeaturesTest)
predictionPolyTrain = regressionPoly.predict(netEnrollmentPolyFeaturesTrain)

coefficientsPoly = regressionPoly.coef_
meansquarederrorPoly = sklearn.metrics.mean_squared_error(netEnrollmentResponseTest, predictionPolyTest)
rPolyTest2 = sklearn.metrics.r2_score(netEnrollmentResponseTest, predictionPolyTest)
rPolyTrain2 = sklearn.metrics.r2_score(netEnrollmentResponseTrain, predictionPolyTrain)

# Very clearly overfitting (train .999, test negative). Trying ridge regression instead

regressionPoly = linear_model.Ridge()
regressionPoly.fit(netEnrollmentPolyFeaturesTrain, netEnrollmentResponseTrain)

# Make predictions
predictionPolyTest = regressionPoly.predict(netEnrollmentPolyFeaturesTest)
predictionPolyTrain = regressionPoly.predict(netEnrollmentPolyFeaturesTrain)

coefficientsPoly = regressionPoly.coef_
meansquarederrorPoly = sklearn.metrics.mean_squared_error(netEnrollmentResponseTest, predictionPolyTest)
rPolyTest2Ridge = sklearn.metrics.r2_score(netEnrollmentResponseTest, predictionPolyTest)
rPolyTrain2Ridge = sklearn.metrics.r2_score(netEnrollmentResponseTrain, predictionPolyTrain)

# Doesn't perform any better

# not used
##### HISTOGRAMS
# 5 gdp per capita
# 6 gni per capita
# 7 internet users
# 16 youngPopRatio???

'''
np.histogram(validCountries[:,1])
np.histogram(validCountries[:,2])
np.histogram(validCountries[:,3])
'''

### Matrix completion algorithm
matrix = np.array(validCountries)[:,5:].astype(float)


# built in functions didn't like nan values, so I created a couple functions
# that could work around this issue

# this takes the two columns that I want to find the correlation between and 
# only compares values where they exist in both columns. That way any missing
# entries are ignored
def correlation(columns):
    indices = []
    for i in range(0, len(columns)):
        if np.isnan(columns[i,0]) == False:
            if np.isnan(columns[i,1]) == False:
                indices.append(i)
    return(np.corrcoef(columns[[indices]], rowvar = False)[0,1])

# creating a matrix showing all the correlations, easy to see which variables
# are correlated with which
def corrMatrix(matrix):
    maximum = len(np.transpose(matrix))
    corrMatrix = np.zeros([maximum,maximum])
    for i in range(0, maximum):
        for j in range(0, maximum):
            if i == j:
                corrMatrix[i,j] = 1
            else:
                corrMatrix[i,j] = correlation(np.array(matrix[:,[i,j]]))
    return corrMatrix

# using a stochastic gradient descent algorithm to approximate
# P and Q such that PQ = R wherever R has a value
def matrix_completion(R, K = 4, steps = 2000, alpha = .0002, beta = .002):
    [iMax, jMax] = R.shape
    P = np.random.rand(iMax, K)
    Q = np.random.rand(K, jMax)
    # making empty entries 0
    for i in range(0, iMax):
        for j in range(0, jMax):
            if np.isnan(R[i,j]):
                R[i,j] = 0
    while (steps > 0):
        steps = steps - 1
        # keeping track of the error
        errorTrace = 0
        count = 0
        # looping over every entry of R
        for i in range(0, iMax):
            for j in range(0, jMax):
                if R[i,j] != 0:
                    # finding the error
                    error = R[i,j] - np.matmul(Q.transpose()[j], P[i])
                    errorTrace += error
                    count += 1
                    # adjusting P and Q 
                    P[i] = P[i] + beta * (error * Q[:,j] - alpha * P[i])
                    Q[:,j] = Q[:,j] + beta * (error * P[i] - alpha * Q[:,j])
                    
    nP = P[:]
    nQ = Q[:]    
    print('Done')
    return(np.matmul(nP,nQ), errorTrace/count)


error = [0,0]
# splitting the data into two groups based on the correlations (each group is
# correlated to other variables in the group, but not to the other group)
# GDP per capita, GNI per capita and internet users per 100 are correlated
firstFeatures = np.array(validCountries)[:,[5,6,7]].astype(float)
completedFirstFeatures, error[0] = matrix_completion(firstFeatures/1e6, K = 2)

# Population features
secondFeatures = np.array(validCountries)[:,[8,9,10,11,12,13]].astype(float)
completedSecondFeatures, error[1] = matrix_completion(secondFeatures/1e6, K = 3, steps = 5000)

# Put them back together
completedFeatures = np.hstack((completedFirstFeatures * 1e6, 
                               completedSecondFeatures * 1e6))




# Split into training and test
completedFeaturesTrain = completedFeatures[(train,[0,1,2,3,4,5,6,7,8])]
completedFeaturesTest = completedFeatures[(test,[0,1,2,3,4,5,6,7,8])]

# Trying with this new data

# model training
regression = linear_model.LinearRegression()
regression.fit(completedFeaturesTrain, netEnrollmentResponseTrain)

# predictions
prediction = regression.predict(completedFeaturesTest)
predictionTrain = regression.predict(completedFeaturesTrain)

rTestC = sklearn.metrics.r2_score(netEnrollmentResponseTest, prediction)
rTrainC = sklearn.metrics.r2_score(netEnrollmentResponseTrain, predictionTrain)

# trying the same with polynomial 

polyRegression = sklearn.preprocessing.PolynomialFeatures(2)
completedPolyFeaturesTrain = polyRegression.fit_transform(completedFeaturesTrain)
completedPolyFeaturesTest = polyRegression.fit_transform(completedFeaturesTest)

# Linear regression training
regressionPoly = linear_model.LinearRegression()
regressionPoly.fit(completedPolyFeaturesTrain, netEnrollmentResponseTrain)

# Make predictions
predictionPolyTest = regressionPoly.predict(completedPolyFeaturesTest)
predictionPolyTrain = regressionPoly.predict(completedPolyFeaturesTrain)

rPolyTestC = sklearn.metrics.r2_score(netEnrollmentResponseTest, predictionPolyTest)
rPolyTrainC = sklearn.metrics.r2_score(netEnrollmentResponseTrain, predictionPolyTrain)


#### Using PCA to reduce feature dimension ( would be cool if it worked so 
# we could get nice graphic results (since it's low dimensional))
pca = PCA(1, copy = False, random_state = 42)
pca.fit(completedFeaturesTrain)

completedFeaturesTrainPCA = pca.transform(completedFeaturesTrain)
completedFeaturesTestPCA = pca.transform(completedFeaturesTest)

regression = linear_model.LinearRegression()
regression.fit(completedFeaturesTrainPCA, netEnrollmentResponseTrain)

predictionTestPCA = regression.predict(completedFeaturesTestPCA)
predictionTrainPCA = regression.predict(completedFeaturesTrainPCA)

rTestPCA = sklearn.metrics.r2_score(netEnrollmentResponseTest, predictionTestPCA)
rTrainPCA = sklearn.metrics.r2_score(netEnrollmentResponseTrain, predictionTrainPCA)
# doesn't work very well
fig = plt.figure()
plt.plot(completedFeaturesTrainPCA, netEnrollmentResponseTrain, 'ro')
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(completedFeaturesTrainPCA[:,0], completedFeaturesTrainPCA[:,1], netEnrollmentResponseTrain)


completedYoungPopRatioTrain = youngPopulationRatio(completedFeaturesTrain[:,7], completedFeaturesTrain[:,8])
completedYoungPopRatioTest = youngPopulationRatio(completedFeaturesTest[:,7], completedFeaturesTest[:,8])


# doing the same stuff as the first few trials with the values that have been
# filled in by the matrix factorization algorithm
newCompFeatTrain = np.zeros([100, 4])
newCompFeatTest = np.zeros([30, 4])

for i in range(0, 4):
    if i != 3:
        newCompFeatTrain[:,i] = completedFeaturesTrain[:,i]
        newCompFeatTest[:,i] = completedFeaturesTest[:,i]
    else:
        newCompFeatTrain[:,i] = completedYoungPopRatioTrain
        newCompFeatTest[:,i] = completedYoungPopRatioTest
        



netEnrollmentFeaturesTrain = newCompFeatTrain
netEnrollmentFeaturesTest = newCompFeatTest

regression = linear_model.LinearRegression()
regression.fit(netEnrollmentFeaturesTrain, netEnrollmentResponseTrain)

# Make predictions
prediction = regression.predict(netEnrollmentFeaturesTest)
predictionTrain = regression.predict(netEnrollmentFeaturesTrain)

coefficients = regression.coef_
meansquarederror = sklearn.metrics.mean_squared_error(netEnrollmentResponseTest, prediction)
rTest = sklearn.metrics.r2_score(netEnrollmentResponseTest, prediction)
rTrain = sklearn.metrics.r2_score(netEnrollmentResponseTrain, predictionTrain)

# Try standardizing features first

scaler = StandardScaler()
scalerTrain = list(netEnrollmentFeaturesTrain)
scalerTest = list(netEnrollmentFeaturesTest)
scaler.fit(scalerTrain)
netEnrollmentFeaturesTrain2 = scaler.transform(scalerTrain)
netEnrollmentFeaturesTest2 = scaler.transform(scalerTest)

# Linear regression training
regression2 = linear_model.LinearRegression()
regression2.fit(netEnrollmentFeaturesTrain2, netEnrollmentResponseTrain)

# Make predictions
prediction2Test = regression2.predict(netEnrollmentFeaturesTest2)
prediction2Train = regression2.predict(netEnrollmentFeaturesTrain2)


coefficients2 = regression2.coef_
meansquarederror2 = sklearn.metrics.mean_squared_error(netEnrollmentResponseTest, prediction2Test)
r2Test = sklearn.metrics.r2_score(netEnrollmentResponseTest, prediction2Test)
r2Train = sklearn.metrics.r2_score(netEnrollmentResponseTrain, prediction2Train)

# Linear regression doesn't seem to be working well. Maybe we can fit some polynomial
# or log thing here

polyRegression = sklearn.preprocessing.PolynomialFeatures(2)
netEnrollmentPolyFeaturesTrain = polyRegression.fit_transform(netEnrollmentFeaturesTrain)
netEnrollmentPolyFeaturesTest = polyRegression.fit_transform(netEnrollmentFeaturesTest)

# Linear regression training
regressionPoly = linear_model.LinearRegression()
regressionPoly.fit(netEnrollmentPolyFeaturesTrain, netEnrollmentResponseTrain)

# Make predictions
predictionPolyTest = regressionPoly.predict(netEnrollmentPolyFeaturesTest)
predictionPolyTrain = regressionPoly.predict(netEnrollmentPolyFeaturesTrain)

coefficientsPoly = regressionPoly.coef_
meansquarederrorPoly = sklearn.metrics.mean_squared_error(netEnrollmentResponseTest, predictionPolyTest)
rPolyTest = sklearn.metrics.r2_score(netEnrollmentResponseTest, predictionPolyTest)
rPolyTrain = sklearn.metrics.r2_score(netEnrollmentResponseTrain, predictionPolyTrain)

# not as good as the original








