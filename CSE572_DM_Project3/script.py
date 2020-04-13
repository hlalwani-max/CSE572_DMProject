# Just a test script. No work done yet/
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def missingValueMean(data):
    data.dropna(how="all")
    imp = SimpleImputer(strategy="mean")
    return pd.DataFrame(imp.fit_transform(data))


def readFileToPandas(data_file):
    data_file_deli = ','
    largest_column_count = 0

    with open(data_file, 'r') as f:
        lines = f.readlines()

        for line in lines:
            column_count = len(line.split(data_file_deli)) + 1
            largest_column_count = column_count if largest_column_count < column_count else largest_column_count

    f.close()

    column_names = [i for i in range(0, largest_column_count)]
    df = pd.read_csv(data_file, header=None, delimiter=data_file_deli, names=column_names)
    return df


def carbToBins(data):
    # dict = {"b1": [], "b2": [], "b3": [], "b4": [], "b5": []}

    for i in range(len(data)):
        if data.iloc[i, 0] == 0:
            data.iloc[i, 0] = 1
        elif data.iloc[i, 0] in range(1, 21):
            data.iloc[i, 0] = 2
        elif data.iloc[i, 0] in range(21, 41):
            data.iloc[i, 0] = 3
        elif data.iloc[i, 0] in range(41, 61):
            data.iloc[i, 0] = 4
        elif data.iloc[i, 0] in range(61, 81):
            data.iloc[i, 0] = 5
        else:
            data.iloc[i, 0] = 6


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


def getMealLabelledData():

    meal1 = readFileToPandas("MealNoMealData/mealData1.csv")
    meal2 = readFileToPandas("MealNoMealData/mealData2.csv")
    meal3 = readFileToPandas("MealNoMealData/mealData3.csv")
    meal4 = readFileToPandas("MealNoMealData/mealData4.csv")
    meal5 = readFileToPandas("MealNoMealData/mealData5.csv")

    meal1 = missingValueMean(meal1)
    meal2 = missingValueMean(meal2)
    meal3 = missingValueMean(meal3)
    meal4 = missingValueMean(meal4)
    meal5 = missingValueMean(meal5)

    meal = pd.concat([meal1.iloc[0:50, 0:30], meal2.iloc[0:50, 0:30], meal3.iloc[0:50, 0:30], meal4.iloc[:50, 0:30],
                      meal5.iloc[0:50, 0:30]])

    meal_label = carbToBinsConcat()

    data = meal.copy()
    data[30] = np.nan

    for i in range(len(meal)):
        data.iloc[i, 30] = meal_label.iloc[i,0]
    return data

data = getMealLabelledData()
print(data)
