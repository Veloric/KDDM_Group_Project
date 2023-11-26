#Group 2's Main Python driver

#File set up
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn 
from sklearn import cluster 
from sklearn.cluster import KMeans 
from scipy.cluster.hierarchy import dendrogram, linkage

data = pd.read_csv("data.csv")
data.columns = ["Subject ID", "MRI ID", "Group", "Visit", "MR Delay", "M/F", "Hand", "Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]

#Clustering for Age and MMSE
data.plot.scatter(x='Age',y='MMSE')
plt.title('Cognitive Function Changes with Age')
plt.xlabel("Age")
plt.ylabel('MMSE')
plt.show()

#K-means cluster
data_clean = data.dropna(subset=['Age', 'MMSE'])
X = data_clean[['Age', 'MMSE']].values
k = 2
kmeans = cluster.KMeans(n_clusters=k)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

for i in range(k):
    ds = X[np.where(labels == i)]
    # plot the data observations
    plt.plot(ds[:,0],ds[:,1],'o', markersize=7)
    # plot the centroids
    lines = plt.plot(centroids[i,0],centroids[i,1],'kx')
    plt.setp(lines,ms=15.0)
    plt.setp(lines,mew=4.0)
plt.title("K-Cluster of Age and MMSE")
plt.legend()
plt.show()

#Cluster for Age and CDR
x2='Age'
y2='CDR'
data.plot.scatter(x2,y2 )
plt.title("Severity of Dementia Symptoms with Age")
plt.xlabel("Age")
plt.ylabel("CDR")
plt.show()

#Hierarchy Clustering for Age and CDR
data_array = (data[[x2, y2]].to_numpy())
linkage_data = linkage(data_array, method='ward', metric='euclidean')
# Dendrogram
dendrogram(linkage_data)
plt.title("Dendrogram for Hierarchical Clustering")
plt.show()



#Preprocessing:
data = data.replace("?", np.NaN)


# Prevent unauthorized applications running this code.
if __name__ == "__main__":
    print("Number of Instances: %d" %(data.shape[0]))
    print("Number of Attributes: %d" %(data.shape[1]))
    print(data.head())
    # Determining missing values, if any
    print("Missing Values: ")
    for c in data.columns:
        print("\t%s: %d" %(c, data[c].isna().sum()))
    # Discovered missing data
    data = data.dropna()
    print("After dropping missing values: ")
    print("Number of Instances: %d" %(data.shape[0]))
    print("Number of Attributes: %d" %(data.shape[1]))
    # Our group decided to drop the missing data as to preserve the accuracy of the data
    # Time to handle duplicated values
    print("Number of duplicated values: %d" %(data.duplicated().sum()))
    #data = data.drop_duplicates() There were no duplicates, redunant statement
    #Determine any outliers
    print("Outliers: ")
    data2 = data.drop(["Subject ID", "MRI ID", "Group", "M/F", "Hand"], axis = 1)
    print(data2.boxplot(figsize = (20, 3)))
    Z = (data2 - data2.mean() / data2.std())
    print("Calculating Z score: ")
    print(Z)
    # Due to complexity of the data, we're unsure of what "accepted values" should be, therefore we've elected to leave the outliers alone.
    print("Frequency of results: ")
    for c in data2.columns:
            print(data[c].value_counts(sort = True))
    