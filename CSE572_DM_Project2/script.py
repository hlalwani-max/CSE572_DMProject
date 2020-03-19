import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import pickle, math
import scipy.stats as stat
import scipy.fftpack as fftp


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
    # m_fft = np.zeros((data.shape[0], data.shape[1]))
    # for i in range(data.shape[0]):
    #     m_fft[i, :] = fftp.fft(data.iloc[i, :])
    # m_fft = np.zeros((data.shape[0], 1))
    # m_fft = data.apply(lambda row: row.fftp.fft(data), axis=1)
    m_fft = fftp.fft(data, axis=0)
    np.absolute(m_fft)
    sorted_m_fft = np.copy(m_fft)
    m_fft.sort()
    FFT_Top5Features = sorted_m_fft[:, ::-1][:, :5]
    return FFT_Top5Features


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
    # print(meal1, meal2, meal3, meal4, meal5)
    meal = pd.concat([meal1, meal2, meal3, meal4, meal5])
    meal.dropna(how="any")
    meal.fillna(method='ffill', inplace=True)
    mealConcat = meal.loc[:, :30]
    mealConcat[30] = 1
    # print("no meal concat:\n", mealConcat)

    nomeal1 = readFileToPandas("MealNoMealData/Nomeal1.csv")
    nomeal2 = readFileToPandas("MealNoMealData/Nomeal2.csv")
    nomeal3 = readFileToPandas("MealNoMealData/Nomeal3.csv")
    nomeal4 = readFileToPandas("MealNoMealData/Nomeal4.csv")
    nomeal5 = readFileToPandas("MealNoMealData/Nomeal5.csv")
    # print(nomeal1, nomeal2, nomeal3, nomeal4, nomeal5)
    nomeal = pd.concat([nomeal1, nomeal2, nomeal3, nomeal4, nomeal5])
    nomeal.dropna(how="any")
    nomeal.fillna(method='ffill', inplace=True)
    nomealConcat = nomeal.loc[:, :30]
    nomealConcat[30] = 0
    # print("no meal concat:\n", nomealConcat)

    meal_nomeal = pd.concat([mealConcat, nomealConcat])

    # Kurtosis
    kurtosis_out = kurtosis(meal_nomeal)
    # print("kurtosis", kurtosis_out)

    # LAGE
    lage_out = lage(meal_nomeal)
    # print("LAGE: ", LAGE_out)

    # Entropy
    entropy_out = entropy(meal_nomeal)
    # print("Entropy:", entropy_out)

    # FFT Top5 features
    fft5_out = mfft(meal_nomeal)
    print("FFT Top5", fft5_out)

    # FeatureMatrix
    featureMAT = np.hstack((fft5_out, kurtosis_out, lage_out, entropy_out))

    # print(meal_nomeal)
    X = meal_nomeal.iloc[:, [i for i in range(30)]]
    y = meal_nomeal.iloc[:, 30]
    # print(X.shape, y.shape)

    # print(X) #Pd dataframe
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    # print(X) #nd numpy

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    pca = PCA(n_components=2)
    pca.fit(X_train)
    X_t_train = pca.transform(X_train)
    X_t_test = pca.transform(X_test)
    clf = SVC()
    clf.fit(X_t_train, y_train)
    predicted = clf.predict(X_t_test)
    scores = clf.score(X_t_test, y_test)
    print('score', scores)
    print('pred label', predicted)

    saved_model = pickle.dumps(clf)
    return saved_model, scaler


model_load, sc = model()

# data = readFileToPandas("test.csv")
data = pd.read_csv("test.csv", header=None, keep_default_na=False)
data.dropna(how="all")
# data.replace(to_replace=float('nan'), value=data.mean(axis = 0), inplace=True)
# print(math.isnan(data[30]))
# data.interpolate(method='polynomial', inplace=True, order = 2)
data.fillna(method='ffill', inplace=True)

# X_ttest = data.loc[:, :30]
X_ttest = data
print(X_ttest.shape)
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

