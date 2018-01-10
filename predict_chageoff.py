import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import random as rd

################################################################
#Prepare the part of the test set that includes the individuals that have not charged off so far
#Then save it in a csv file so as not to rerun it over and over again
#These values are random numnbers from 1 to their checked-in values

#def fillRandom(col1,col2,dtFrame):
#    for ii in range(len(dtFrame)):
#        if dtFrame[col1][ii]>1:
#            number=rd.randint(1,dtFrame[col1][ii])
#        else:
#            number=1
#        print (ii, number)
#        dtFrame[col2][ii]=number

#fillRandom('fromOrigination','fromOriginationtoChargeoff',dfTest)
#dfTest.to_csv('TestSet')
#######################################################################



#Read in the data
df = pd.read_csv('loan_timing.csv')

#Days in three years
threeYears=3*365
#Rename the columns
df.columns= ['fromOrigination', 'fromOriginationtoChargeoff']

#Prediction percentages of charge-offs
predictionList=[]

#Number of iterations
iterations=10000

for iteration in range(iterations):

    # Include the ones that have charged off in the training set
    #Others are in the test set, to be predicted
    dfTrain= df[pd.isnull(df['fromOriginationtoChargeoff'])==False].copy()
    dfTest= df[pd.isnull(df['fromOriginationtoChargeoff'])].copy()
    dfTrain['chargedOff']=True
    
    #I don't need their index
    dfTrain = dfTrain.reset_index(drop=True)
    dfTest = dfTest.reset_index(drop=True)
    
    # The one that haven't charged off. Read it from the readily prepared csv file. The routine is commented off above.
    dfTest = pd.read_csv('TestSet')
    dfTest['chargedOff']=False
    dfTest.drop(['Unnamed: 0'],axis=1,inplace=True)
    
    
    #First part of the train set. All positive charge-offs
    xTrain=dfTrain[['fromOrigination', 'fromOriginationtoChargeoff']].values.tolist()
    yTrain=dfTrain['chargedOff'].values.tolist()
    
    
    #Second part of the test set. All negative charge-offs
    x_train2, x_test, y_train2, y_test2 = train_test_split(dfTest.drop('chargedOff',axis=1), 
                                                        dfTest['chargedOff'], test_size=0.93)
    
    #Merge both test sets.
    xTrain=xTrain+x_train2.values.tolist()
    yTrain=yTrain+y_train2.values.tolist()
    
            
    #Build the model and train it.
    logmodel = LogisticRegression()
    logmodel.fit(xTrain,yTrain)
    
    #Test the model for the test set. That is for three years. What percentage is going to charge-off?
    dfTest['fromOriginationtoChargeoff']=threeYears
    dfTest.drop('chargedOff',axis=1,inplace=True)
        
    xTest=dfTest[['fromOrigination', 'fromOriginationtoChargeoff']].values.tolist()
    predictions = logmodel.predict(xTest)
    
    
    dfTest['Predicted']=predictions
    #Get the True values, meaning that the charge-offs
    dfTest['Predicted'].value_counts()
    
    #If no charge-offs, not a single one; disregard it
    try:
        predictionList.append(dfTest['Predicted'].value_counts()[1]/len(dfTest))
    except:
        print('No charge offs')

#Print the prediction, the average of all charge-off percentages
print("Average percentage of charge-offs", sum(predictionList)/len(predictionList))
