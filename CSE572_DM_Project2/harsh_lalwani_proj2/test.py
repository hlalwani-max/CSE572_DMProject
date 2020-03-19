import joblib
import pandas as pd
import train
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

train.model()

mod = joblib.load("model.pkl")

test_data = pd.read_csv("test.csv", header=None)
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

predicted = mod.predict(X_ttest)
print("Predicted:", predicted)
