import pickle

import numpy as np
import pandas as pd
import scipy.fftpack as fftp
import scipy.stats as stat
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
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


# def fillNaN(df):
#     m = df.mean(axis=1)
#     for i, col in enumerate(df):
#         # using i allows for duplicate columns
#         # inplace *may* not always work here, so IMO the next line is preferred
#         # df.iloc[:, i].fillna(m, inplace=True)
#         df.iloc[:, i] = df.iloc[:, i].fillna(m)
#         return df

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
    # meal.fillna(method='ffill', inplace=True)
    print("Nan", np.where(np.isnan(meal)))
    # meal = fillNaN(meal)
    meal = missingValueMean(meal)
    mealConcat = meal.loc[:, :30]
    mealConcat[30] = 1
    # print("no meal concat:\n", mealConcat)

    nomeal = pd.concat([nomeal1, nomeal2, nomeal3, nomeal4, nomeal5])
    nomeal.dropna(how="all")
    # nomeal.fillna(method='ffill', inplace=True)
    # nomeal.fillna(nomeal.mean(axis=1), inplace=True)
    print(" Nan", np.where(np.isnan(meal)))
    nomeal = missingValueMean(nomeal)
    nomealConcat = nomeal.loc[:, :30]
    nomealConcat[30] = 0
    # print("no meal concat:\n", nomealConcat)

    meal_nomeal = pd.concat([mealConcat, nomealConcat])
    f_mat = featureMat(meal_nomeal)

    pca = PCA(n_components=5)
    pca_fit = pca.fit_transform(f_mat)
    # print("PCA shape:", pca_fit.shape)
    # print("PCA:", pca_fit)

    X = pca_fit
    y = meal_nomeal.iloc[:, 30]

    '''
    # print(meal_nomeal)
    X = meal_nomeal.iloc[:, [i for i in range(30)]]
    y = meal_nomeal.iloc[:, 30]
    # print(X.shape, y.shape)
    '''

    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    # print(X) #nd numpy

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf = SVC()
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    scores = clf.score(X_test, y_test)
    print('score', scores)
    print('pred label', predicted)

    saved_model = pickle.dumps(clf)
    return saved_model, scaler


model_load, sc = model()

# data = readFileToPandas("test.csv")
test_data = pd.read_csv("test.csv", header=None, keep_default_na=False)
# test_data.dropna(how="all")
# test_data.replace(to_replace=float('nan'), value=test_data.mean(axis = 0), inplace=True)
# print(math.isnan(test_data[30]))
# test_data.interpolate(method='polynomial', inplace=True, order = 2)
test_data.fillna(method='ffill', inplace=True)

# X_ttest = test_data.loc[:, :30]
X_ttest = featureMat(test_data)
# print(X_ttest.shape)
# print(max(X_test.values))

if len(X_ttest) > 1:
    X_ttest_minmax = sc.fit_transform(X_ttest)
else:
    X_ttest_minmax = sc.fit_transform(X_ttest.T)

print(X_ttest_minmax)
pca = PCA(n_components=2, svd_solver='full')
pca.fit(X_ttest_minmax)
X_t_ttest = pca.transform(X_ttest_minmax)
# X_t_test = X_test
# print(X_t_test)
svc_from_pickle = pickle.loads(model_load)
# print(X_t_test.shape)
predicted = svc_from_pickle.predict(X_t_ttest)
print("Predicted:", predicted)
