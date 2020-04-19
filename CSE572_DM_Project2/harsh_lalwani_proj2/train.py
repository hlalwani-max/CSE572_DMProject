import pickle
import statistics

import joblib
import numpy as np
import pandas as pd
import scipy.fftpack as fftp
import scipy.stats as stat
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler as ss
from sklearn.svm import SVC


def kurtosis(data):
    m_Kurtosis = np.zeros((data.shape[0], 1))
    m_Kurtosis[:, 0] = data.apply(lambda row: row.kurtosis(), axis=1)
    return m_Kurtosis


def lage(data):
    m_LAGE = np.zeros((data.shape[0], 1))
    m_LAGE[:, 0] = data.apply(lambda row: row.max() - row.min(), axis=1)
    return m_LAGE


def entropy(data):
    m_entropy = np.zeros((data.shape[0], 1))
    for i in range(data.shape[0]):
        m_entropy[i, :] = stat.entropy(data.iloc[i, :])
    return m_entropy


# FFT Top 5 Features shape (216,5)
def mfft(data):
    m_fft = fftp.fft(data, axis=0)
    np.absolute(m_fft)
    sorted_m_fft = np.copy(m_fft)
    m_fft.sort()
    FFT_Top5Features = sorted_m_fft[:, ::-1][:, :5]
    return FFT_Top5Features


def featureMat(data):
    # Kurtosis
    kurtosis_out = kurtosis(data)
    # print("kurtosis", kurtosis_out)

    # LAGE
    lage_out = lage(data)
    # print("LAGE: ", LAGE_out)

    # Entropy
    entropy_out = entropy(data)
    # print("Entropy:", entropy_out)

    # FFT Top5 features
    fft5_out = mfft(data)
    # print("FFT Top5", fft5_out)

    # FeatureMatrix
    featureMAT = np.hstack((fft5_out, kurtosis_out, lage_out, entropy_out))

    featureMAT = ss().fit_transform(featureMAT.real)

    return featureMAT


def missingValueMean(data):
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


def model():
    meal1 = readFileToPandas("MealNoMealData/mealData1.csv")
    meal2 = readFileToPandas("MealNoMealData/mealData2.csv")
    meal3 = readFileToPandas("MealNoMealData/mealData3.csv")
    meal4 = readFileToPandas("MealNoMealData/mealData4.csv")
    meal5 = readFileToPandas("MealNoMealData/mealData5.csv")
    nomeal1 = readFileToPandas("MealNoMealData/Nomeal1.csv")
    nomeal2 = readFileToPandas("MealNoMealData/Nomeal2.csv")
    nomeal3 = readFileToPandas("MealNoMealData/Nomeal3.csv")
    nomeal4 = readFileToPandas("MealNoMealData/Nomeal4.csv")
    nomeal5 = readFileToPandas("MealNoMealData/Nomeal5.csv")
    meal = pd.concat([meal1, meal2, meal3, meal4, meal5])

    meal.dropna(how="all")
    meal = missingValueMean(meal)
    mealConcat = meal.loc[:, :30]
    mealConcat[30] = 1

    nomeal = pd.concat([nomeal1, nomeal2, nomeal3, nomeal4, nomeal5])
    nomeal.dropna(how="all")
    nomeal = missingValueMean(nomeal)
    nomealConcat = nomeal.loc[:, :30]
    nomealConcat[30] = 0

    meal_nomeal = pd.concat([mealConcat, nomealConcat])
    f_mat = featureMat(meal_nomeal)

    pca = PCA(n_components=8)
    pca_fit = pca.fit_transform(f_mat)

    X = pca_fit
    y = meal_nomeal.iloc[:, 30]

    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    y = np.array(y)

    splits = 5
    scores_rf = []
    kf = model_selection.KFold(n_splits=splits, shuffle=True, random_state=None)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        rf = RandomForestClassifier(max_depth=4, random_state=0)
        rf.fit(X_train, y_train)
        scores_rf.append(rf.score(X, y))
    acc_rf = statistics.mean(scores_rf)
    print("RF accuracy = ", acc_rf)

    scores_svm = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        svm = SVC()
        svm.fit(X_train, y_train)
        scores_svm.append(svm.score(X_test, y_test))
    acc_svm = statistics.mean(scores_svm)
    print("SVM accuracy = ", acc_svm)

    rf_file = "rf.pkl"
    joblib.dump(rf, rf_file)
    svm_file = "svm.pkl"
    joblib.dump(svm, svm_file)

