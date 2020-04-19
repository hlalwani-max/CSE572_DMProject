import joblib
import pandas as pd
import train
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import csv
import numpy as np

train.model()

rf_mod = joblib.load("rf.pkl")
svm_mod = joblib.load("svm.pkl")

test_data = pd.read_csv("TestData.csv", header=None)
test_data = train.missingValueMean(test_data)

X_ttest = train.featureMat(test_data)

sc = MinMaxScaler(feature_range=(0, 1))
if len(X_ttest) > 1:
    X_ttest = sc.fit_transform(X_ttest)
else:
    X_ttest = sc.fit_transform(X_ttest.T)

pca = PCA(n_components=8)
pca.fit(X_ttest)
X_ttest_pca = pca.transform(X_ttest)


rf_predicted = rf_mod.predict(X_ttest)
print("Total Test cases:", len(rf_predicted))
print("RF Predicted:", rf_predicted)

svm_predicted = svm_mod.predict(X_ttest)
print("SVM Predicted:", svm_predicted)

#saveToFile
np.savetxt("TestOutput.csv", rf_predicted, delimiter=",", fmt='%d')


