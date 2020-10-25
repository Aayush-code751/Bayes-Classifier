#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


import pandas as pd
import numpy as np
import math
import random


# In[2]:


# loading taining data
def load_train_data():
    traindata=pd.read_csv("D:/550_pattern_recog/Project1Data/data/train.csv", sep=',', names=['A','b','c','d','e','f','g','h','i'])
    return traindata


# In[3]:


#loading test data
def loadFile():
    myDataset= pd.read_csv("D:/550_pattern_recog/Project1Data/data/test.csv", sep=',', names=['A','b','c','d','e','f','g','h','i'])
    return myDataset


# In[5]:


# dataframe=loadFile()
# d=(dataframe.iloc[:,-1]).tolist()
# print(d)
# dt=dataframe.values.tolist()
# print(dt)
# print(dataframe)
# no_diabetes=dataframe[dataframe.iloc[:,-1]==0]
# print(no_diabetes)
# dataframe=dataframe.drop(dataframe.columns[-1],axis=1)
# print(dataframe)


# In[6]:


# calculate standarddeviation from datarame and return a list
def std_dev(dataset):
    stddev_var=dataset.std(axis=0)
    stddevlist=stddev_var.tolist()
    return stddevlist


# In[7]:


# calculate mean from datarame and return a list
def mean_func(dataset):
    mean_var=dataset.mean(axis=0)
    meanlist= mean_var.tolist()
    return meanlist


# In[8]:


def categorize(dataframe):
    #     no_diabetes is the dataframe with last column having values as 0, diabetes is dataframe with last column having values 1
    no_diabetes=dataframe[dataframe.iloc[:,-1]==0]
    diabetes=dataframe[dataframe.iloc[:,-1]==1]

    #     calculating mean and std dev of nodiabetics and diabetics
    mean_ofNoDiabetics = mean_func(no_diabetes)
    mean_ofDiabetics   =mean_func(diabetes)
    std_devofNoDiabetics=std_dev(no_diabetes)
    std_devofDiabetics=std_dev(diabetes)
    

    # deleting the last column(last values fron the list) as we dont want it    
    del mean_ofNoDiabetics[-1]
    del mean_ofDiabetics[-1]
    del std_devofNoDiabetics[-1]
    del std_devofDiabetics[-1]
    
    nodiablist=[]
    diablist=[]
    

    #    appending mean and std dev of  particular attributes(columns) according to their class i.e. diabetic or nodiabetic
    for (a, b,c,d) in zip(mean_ofNoDiabetics,std_devofNoDiabetics,mean_ofDiabetics,std_devofDiabetics):
                   nodiablist.append((a,b))
                   diablist.append((c,d))

    #mapping 0 as mean and std dev of nodiabetics,1 as mean and std dev of diabetics
    category={
        0:nodiablist,
        1:diablist
              }
    return category


# In[9]:


# dataframe=loadFile()
# print(categorize(dataframe))


# In[10]:


def normalDistribution(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom


# In[11]:


# pbb of attributes belonging to class 0 or 1
def classProbablities(categorize,inpu_t):
    probablities={0:1,1:1}  

    #  key is the class name i.e. 0 or 1 and values are the tuples of mean and std dev per attribute.
    #  in our case number of values will be 8 eacjh class and input is actually the firstrow of test data.
    #  we calculate normal distribution for the attribute according to the mean and std dev of that attribute accrding to trainingdata.
    #  accordingly we calculate probablity of that row  wrt to 0 and 1 and store it in probablities dictionary.   
    for key,values in categorize.items():
        for i in range(len(values)):
            mean,stddev=values[i]
            x=inpu_t[i]
            if (key==0):
                probablities[0] *=normalDistribution(x,mean,stddev) 
            else:    
                probablities[1] *=normalDistribution(x,mean,stddev) 
    return probablities           


# In[12]:


# dataframe=loadFile()
# cat=categorize(dataframe)
# inputl=[1.1,2,3,4,5,6,7,8]
# pbb=classProbablities(cat,inputl)
# print(pbb)


# In[13]:


# from the probablity obtained wrt 0 and 1,this functio returns 1 if the probablity is greater for 1 and vice versa 
def predict(categorize,inpu_t):
    probablities=classProbablities(categorize,inpu_t)
    x=probablities[0]
    y=probablities[1]
    if x>y:
        return 0
    else:
        return 1
    


# In[14]:


# dataframe=loadFile()
# cat=categorize(dataframe)
# inputl=[1.1,2,3,4,5,6,7,8]
# pbb=predict(cat,inputl)
# print(pbb)


# In[15]:


# predicting row wise
def getPredictions(categorize,testset):
    predictions=[]
    for i in range(len(testset)):
        ans = predict(categorize, testset[i])
        predictions.append(ans)
    return predictions    


# In[16]:


# dataframe=loadFile()
# cat=categorize(dataframe)
# inputl=[[1.1,2,3,4,5,6,7,8],[1.1,2,3,4,5,6,7,8]]
# pbb=getPredictions(cat,inputl)
# print(pbb)


# In[17]:


def Accuracy(testset,predictions):
    correct=0
    total=len(testset)
    #comparing last column of the test set with the prediction list obtained fron getPrediction() method  and calculating the accuracy   
    for i in range(total):
        if (testset[i][-1]== predictions[i]):
            correct= correct + 1
    return(correct/float(total)) * 100.0        
    
    


# In[18]:


def main():
    trainset= load_train_data()
    testset = loadFile()
    
    testset_with_lastcol=testset.values.tolist()
    testset_without_lastcol = (testset.drop(testset.columns[-1],axis=1)).values.tolist()

    #categorizing according to diabetics and nodiabetics 
    categories=categorize(trainset)

    #making predictions     
    predictions= getPredictions(categories,testset_without_lastcol)

    #getting accuracy     
    accuracy=Accuracy(testset_with_lastcol,predictions)
    
    
    testset_with_only_lastcolumn=(testset.iloc[:,-1]).tolist()
    
    TN=0
    FP=0
    FN=0
    TP=0

    #calculating TN,FP,FN,TP     
    for (i,j) in zip(predictions,testset_with_only_lastcolumn):
        if i==0 and j==0:
            TN +=1
        elif i==0 and j==1:
            FN +=1
        elif i==1 and j==0:
            FP +=1
        else:
            TP +=1
          
    
    
    
    confusionmatrix_accuracy= (TP + TN)/(TP+FP+TN+FN)
    
    error=(FP +FN)/(TP+FP+TN+FN)
    
    sensitivity=TP/(FN+TP)
    
    specificity=TN/(TN+FP)
    
    print("TN= ",TN)
    print("FN= ",FN)
    print("FP= ",FP)
    print("TP= ",TP)
    
    print("accuracy= ",accuracy)
    
    print("confusionmatrix_accuracy= ",confusionmatrix_accuracy* 100)
    
    print("error= ",error* 100)
    
    print("sensitivity= ",sensitivity* 100)
    
    print("specificity= ",specificity* 100)
#     print("predictions= "predictions)
#     print(testset_with_only_lastcolumn)
main()


# In[ ]:





# In[ ]:




