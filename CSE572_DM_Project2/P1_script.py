import numpy as np
import pandas, scipy.fftpack, scipy.stats, sklearn.preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#Reading and Preprocessing Data
meal1=pandas.read_csv('MealNoMealData/mealData1.csv')
meal1 = meal1.fillna(axis=1, limit = 5)
#meal1.replace(to_replace = np.nan, value = CGMSeriesLunchPatient1.mean(), inplace= True)
CGMSeriesLunchPatient2=pandas.read_csv('DataFolder/CGMSeriesLunchPat2.csv')
CGMSeriesLunchPatient2.replace(to_replace = np.nan, value = CGMSeriesLunchPatient2.mean(), inplace= True)
CGMSeriesLunchPatient3=pandas.read_csv('DataFolder/CGMSeriesLunchPat3.csv')
CGMSeriesLunchPatient3.replace(to_replace = np.nan, value = CGMSeriesLunchPatient3.mean(), inplace= True)
CGMSeriesLunchPatient4=pandas.read_csv('DataFolder/CGMSeriesLunchPat4.csv')
CGMSeriesLunchPatient4.replace(to_replace = np.nan, value = CGMSeriesLunchPatient4.mean(), inplace= True)
CGMSeriesLunchPatient5=pandas.read_csv('DataFolder/CGMSeriesLunchPat5.csv')
CGMSeriesLunchPatient5.replace(to_replace = np.nan, value = CGMSeriesLunchPatient5.mean(), inplace= True)
CGMSeriesLunchPatient = pandas.concat([CGMSeriesLunchPatient1, CGMSeriesLunchPatient2, CGMSeriesLunchPatient3, CGMSeriesLunchPatient4, CGMSeriesLunchPatient5])
#Consider only 30 values from a row
seriesNo = 30
# timeSeries = range
timeSeriesConcat = range(216)
CGMSeriesLunchPatientConcat=CGMSeriesLunchPatient.iloc[:,0:seriesNo]
# print(CGMSeriesLunchPatientConcat)

#Entropy
CGM_entropy = np.zeros((CGMSeriesLunchPatientConcat.shape[0],1))
for i in range(CGMSeriesLunchPatientConcat.shape[0]):
  CGM_entropy[i,:] = scipy.stats.entropy(CGMSeriesLunchPatientConcat.iloc[i,:])

plt.figure()
plt.plot(range(216),CGM_entropy, c= "green")
plt.title("Entropy")
plt.xlabel("Series")
plt.ylabel("Entropy")

#Kurtosis
CGM_Kurtosis = np.zeros((CGMSeriesLunchPatientConcat.shape[0],1))
CGM_Kurtosis[:,0] = CGMSeriesLunchPatientConcat.apply(lambda row: row.kurtosis(),axis = 1)

plt.figure()
plt.plot(range(216),CGM_Kurtosis, c= "green")
plt.title("Kurtosis")
plt.xlabel("Series No.")
plt.ylabel("Kurtosis Values")
# print("Kurtosis Value\n", CGM_Kurtosis)

#Large Amplitude (LAGE)
CGM_LAGE = np.zeros((CGMSeriesLunchPatientConcat.shape[0],1))
CGM_LAGE[:,0] = CGMSeriesLunchPatientConcat.apply(lambda row: row.max() - row.min(),axis = 1)

plt.figure()
plt.plot(range(216),CGM_LAGE, c= "green")
plt.title("Large Amplitude")
plt.xlabel("Series No.")
plt.ylabel("LAGE Values")

#FFT Top 5 Features shape (216,5)
CGM_FFT =scipy.fftpack.fft(CGMSeriesLunchPatientConcat)
np.absolute(CGM_FFT)
sorted_CGM_FFT = np.copy(CGM_FFT)
CGM_FFT.sort()
FFT_Top5Features = sorted_CGM_FFT[:,::-1][:,:5]
# print("Top 5 FFT Features:\n", FFT_Top5Features)

plt.figure()
plt.plot(range(216),np.absolute(CGM_FFT[:,0].real), c= "green")
plt.title("FFT")
plt.xlabel("Series No.")
plt.ylabel("FFT Values")



#Feature Matrix
featureMAT=np.hstack((FFT_Top5Features, CGM_Kurtosis, CGM_LAGE, CGM_entropy))

#PCA
featureMAT = sklearn.preprocessing.StandardScaler().fit_transform(featureMAT.real)
pca = PCA(n_components=5)
pca_fit = pca.fit_transform(featureMAT)
print(pca_fit.shape)
pcaDf = pandas.DataFrame(data = pca_fit, columns = ['c_1', 'c_2','c_3','c_4','c_5'])
#print(principalDf)

#PCA component weights
print("PCA compenent weights:\n", np.absolute(pca.components_))

print("PCA component ratio:\n", pca.explained_variance_ratio_)

# PCA graph plotting

#PCA1
plt.figure()
plt.scatter(range(216),pcaDf['c_1'], c="red")
plt.title("PCA 1")
plt.xlabel("Series")
plt.ylabel("Component 1")

#PCA2
plt.figure()
plt.scatter(range(216),pcaDf['c_2'], c="red")
plt.title("PCA 2")
plt.xlabel("Series")
plt.ylabel("Component 2")

#PCA3
plt.figure()
plt.scatter(range(216),pcaDf['c_3'], c="red")
plt.title("PCA 3")
plt.xlabel("Series")
plt.ylabel("Component 3")

#PCA4
plt.figure()
plt.scatter(range(216),pcaDf['c_4'], c="red")
plt.title("PCA 4")
plt.xlabel("Series")
plt.ylabel("Component 4")

#PCA5
plt.figure()
plt.scatter(range(216),pcaDf['c_5'], c="red")
plt.title("PCA 5")
plt.xlabel("Series")
plt.ylabel("Component 5")

# #Mean Crossing
# CGM_TotalMeanCrossing=np.zeros((CGMSeriesLunchPatientConcat.shape[0],1))
# for i, meal in enumerate(CGMSeriesLunchPatientConcat):
#   count = 0
#   median = CGMSeriesLunchPatientConcat.iloc[i,:].median()
#   for val in CGMSeriesLunchPatientConcat.iloc[i,:]:
#     if val > median:
#       count+=1
#   CGM_TotalMeanCrossing[i][0] = count
# plt.figure()
# plt.plot(range(216),CGM_TotalMeanCrossing, c= "green")
# plt.title("CGM Total Mean Crossing No")
# plt.xlabel("Series No.")
# plt.ylabel("Number of Times Mean Crossing")