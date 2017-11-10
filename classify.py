import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
import time
from sklearn.tree import DecisionTreeClassifier

x=int(input("Give the index of the test image you would like to predict : "))
#Read from train.csv
data=pd.read_csv("train.csv").as_matrix()

#Declare a Decision Tree Classifier Object
clf = DecisionTreeClassifier()

#training data
xtrain=data[20000:,1:]
train_label=data[20000:,0]

#testing data
xtest=data[0:21000,1:]
actual_label=data[0:21000:,0]

#training the classifier
print("training classifier.....")
clf.fit(xtrain,train_label)

#Calculating accuracy
print("Predicting the image at index %d ....."%x)
time.sleep(2)
p=clf.predict(xtest)
count=0
for i in range(0,21000):
    if p[i]==actual_label[i]:
        count+=1

#sample test data (csv-2) , We now predict an image
d=xtest[x]
d.shape=(28,28)
pt.imshow(255-d,cmap='gray')
print("With a Probab of : ",(count/21000)*100)
print("The Number is :",clf.predict( [xtest[x]] ))
pt.show()
