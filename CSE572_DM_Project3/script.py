#Just a test script. No work done yet/
import pandas as pd

def carbToBins(data):
    # dict = {"b1": [], "b2": [], "b3": [], "b4": [], "b5": []}

    for i in range(len(data)):
        if data.iloc[i,0] == 0:
            data.iloc[i, 0] = 1
        elif data.iloc[i,0] in range(1, 21):
            data.iloc[i,0] = 2
        elif data.iloc[i,0] in range(21,41):
            data.iloc[i,0] = 3
        elif data.iloc[i,0] in range(41,61):
            data.iloc[i,0] = 4
        elif data.iloc[i,0] in range(61,81):
            data.iloc[i,0] = 5
        else:
            data.iloc[i,0] = 6

def carbToBinsConcat():
    label_meal1 = pd.read_csv("MealAmountData/mealAmountData1.csv", header=None)
    label_meal2 = pd.read_csv("MealAmountData/mealAmountData2.csv", header=None)
    label_meal3 = pd.read_csv("MealAmountData/mealAmountData3.csv", header=None)
    label_meal4 = pd.read_csv("MealAmountData/mealAmountData4.csv", header=None)
    label_meal5 = pd.read_csv("MealAmountData/mealAmountData5.csv", header=None)

    label_meal1 = label_meal1[:50]
    label_meal2 = label_meal2[:50]
    label_meal3 = label_meal3[:50]
    label_meal4 = label_meal4[:50]
    label_meal5 = label_meal5[:50]

    carbToBins(label_meal1)
    carbToBins(label_meal2)
    carbToBins(label_meal3)
    carbToBins(label_meal4)
    carbToBins(label_meal5)

    label_meal = pd.concat([label_meal1, label_meal2, label_meal3, label_meal4, label_meal5])
    return label_meal

def mealData():
    pass
meal_label = carbToBinsConcat()
print(meal_label)