from DecisionTree import DecisionTree, DisplayTree, Testing
import pandas as pd
from sklearn.model_selection import train_test_split
global root

IrreleventClasses=[]
Dataset = input("Enter Name of Dataset File: ") 
TargetClass = input("Enter Name of Target Class: ")
n = int(input("Enter number of Irrelevent Classes: ")) 
for i in range(0, n): 
    element = input('Enter Irrelevent Class: ')   
    IrreleventClasses.append(element) 
      

fulldata = pd.read_csv(Dataset)
TrainingData,Test=train_test_split(fulldata, test_size=0.2, random_state=42)

   
root=DecisionTree(TrainingData, TargetClass, IrreleventClasses)
print(root.label)

DisplayTree(root)

Testing(Test,root)
