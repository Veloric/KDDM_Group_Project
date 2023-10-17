#Group 2's Main Python driver

#File set up
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("data.csv")
data.columns = ["Subject ID", "MRI ID", "Group", "Visit", "MR Delay", "M/F", "Hand", "Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]

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
    