import pandas as pd, numpy as np
from harsh_lalwani_proj2 import MealNoMealData as dat
from harsh_lalwani_proj2.train import missingValueMean, readFileToPandas, model

def readMealAmountToPandas():
    carb1 = pd.read_csv("mealAmountData1.csv", header=None)
    carb2 = pd.read_csv("mealAmountData2.csv", header=None)
    carb3 = pd.read_csv("mealAmountData3.csv", header=None)
    carb4 = pd.read_csv("mealAmountData4.csv", header=None)
    carb5 = pd.read_csv("mealAmountData5.csv", header=None)
    carb1 = carb1.loc[:49]
    carb2 = carb2.loc[:49]
    carb3 = carb3.loc[:49]
    carb4 = carb4.loc[:49]
    carb5 = carb5.loc[:49]
    carbConcat = pd.concat([carb1, carb2, carb3, carb4, carb5])
    return carbConcat

def carbToBins(data):
    # dict = {"b1": [], "b2": [], "b3": [], "b4": [], "b5": []}

    for i in range(len(data)):
        res = pd.DataFrame()
        # print(item)
        if data.iloc[i,:] in range(21):
            res.iloc[i,:] = 'b1'
        elif data.iloc[i,:] in range(21,41):
            res.iloc[i,:] = 'b2'
        elif data.iloc[i,:] in range(41,61):
            res.iloc[i,:] = 'b3'
        elif data.iloc[0,:] in range(61,81):
            res.iloc[i,:] = 'b4'
        elif data.iloc[i,:] in range(41,61):
            res.iloc[i,:] = 'b5'
        elif data.iloc[i,:] in range(61,81):
            res.iloc[i,:] = 'b6'
        elif data.iloc[i,:] in range(81,101):
            res.iloc[i,:] = 'b7'
        else:
            res.iloc[i,:] = 'b8'
    return res


dat1 = readFileToPandas("harsh_lalwani_proj2/MealNoMealData/mealData1.csv")
dat1.dropna(how = "all")
dat1 = missingValueMean(dat1)
carb = readMealAmountToPandas()
labels = carbToBins(carb)
print(lebels)
# labels = ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"]
# carb_labels = carbToBins(carb, labels)
# print(carb_labels)
dat1 = dat1.loc[:49]
# print("dat1 len:", len(dat1))
# print("carb1:", carb)