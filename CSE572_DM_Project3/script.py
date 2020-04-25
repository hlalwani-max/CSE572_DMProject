import numpy as np
import pandas as pd
import scipy.fftpack as fftp
import scipy.stats as stat
from sklearn import metrics
from sklearn import model_selection
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler as ss
from collections import Counter

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


def dataClusterMapping(data):
    # dict = {"b1": [], "b2": [], "b3": [], "b4": [], "b5": []}

    cluster_mapping_GT = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: []
    }

    cluster_mapping_GT_rev = {}

    for i in range(len(data)):
        if data.iloc[i, 30] == 1:
            cluster_mapping_GT[1].append(i)
            cluster_mapping_GT_rev[i] = 1
        elif data.iloc[i, 30] == 2:
            cluster_mapping_GT[2].append(i)
            cluster_mapping_GT_rev[i] = 2
        elif data.iloc[i, 30] == 3:
            cluster_mapping_GT[3].append(i)
            cluster_mapping_GT_rev[i] = 3
        elif data.iloc[i, 30] == 4:
            cluster_mapping_GT[4].append(i)
            cluster_mapping_GT_rev[i] = 4
        elif data.iloc[i, 30] == 5:
            cluster_mapping_GT[5].append(i)
            cluster_mapping_GT_rev[i] = 5
        else:
            cluster_mapping_GT[6].append(i)
            cluster_mapping_GT_rev[i] = 6

    return cluster_mapping_GT, cluster_mapping_GT_rev


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


def getDataAndLabels():
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

    return meal, meal_label


def getLabelledData(data, labels):
    df = data.copy()
    df[30] = np.nan

    for i in range(len(labels)):
        df.iloc[i, 30] = labels.iloc[i, 0]

    global mapping_data_point_to_cluster

    _, mapping_data_point_to_cluster = dataClusterMapping(df)
    print(mapping_data_point_to_cluster)

    return df


def doKmeans(X, y):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.60, test_size=0.40)

    k_means = KMeans(n_clusters=6)
    k_means.fit(X_train)
    print("\nactual labels:", y_test.flatten())
    Y_pred = k_means.predict(X_test)
    print("Predicted Labels-", Y_pred)
    print("\nCluster centers: ", k_means.cluster_centers_)
    score = metrics.accuracy_score(y_test, Y_pred)
    print('\nAccuracy:{0:f}'.format(score))
    cluster = pd.DataFrame(k_means.labels_)
    print("Old cluster:", cluster)
    clust_to_real_clust_map = mapClusterToCluster(cluster, y_train)

    for i in range(150):
        cluster.iloc[i][0] = clust_to_real_clust_map[cluster.iloc[i][0]]

    print("New cluster with ground truth:", cluster)

    return cluster


def doDBScan(X, y):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.60, test_size=0.40,
                                                                        random_state=101)

    db_scan = DBSCAN(eps=0.265, min_samples=2)
    db_scan.fit(X_train)
    print("\ncluster labels:", db_scan.labels_[:])
    print("\nactual labels:", y_test.flatten())
    db_scan.fit_predict(X_train)
    score = metrics.accuracy_score(y_test, db_scan.fit_predict(X_test))
    print('\nAccuracy:{0:f}'.format(score))
    cluster = pd.DataFrame(db_scan.labels_)

# cluster to correct cluster
def mapClusterToCluster(clusters, ground_label):
    map_data_index_to_clusters = {}
    for i in range(len(clusters)):
        if clusters.iloc[i][0] not in map_data_index_to_clusters:
            map_data_index_to_clusters[clusters.iloc[i][0]] = [ground_label[i][0]]
        else:
            map_data_index_to_clusters[clusters.iloc[i][0]].append(ground_label[i][0])


    map_cluster_to_real_cluster = {}
    for key in map_data_index_to_clusters.keys():
        c = Counter(map_data_index_to_clusters[key]).most_common(1)
        map_cluster_to_real_cluster[key] = c[0][0]

    return map_cluster_to_real_cluster



meal_data, labels = getDataAndLabels()

labelled_meal_data = getLabelledData(meal_data, labels)

f_mat = featureMat(meal_data)

pca = PCA(n_components=8)
pca_fit = pca.fit_transform(f_mat)

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(pca_fit)
y = np.array(labels)

predicted_Kmeans_labels = doKmeans(X, y)
