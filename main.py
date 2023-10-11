#Group 2's Main Python driver

#File set up
import pandas as pd
import numpy as np
data = pd.read_csv("data.csv")
data.columns = ["Subject ID", "MRI ID", "Group", "Visit", "MR Delay", "M/F", "Hand", "Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF"]

#Preprocessing:
data = data.replace("?", np.NaN)


# Prevent unauthorized applications running this code.
if __name__ == "__main__":
    # Determining missing values, if any
    for c in data.columns:
        print("\t%s: %d" %(c, data[c].isna().sum()))
    # Discovered missing data
    data_2 = data["SES"].dropna()
    data_3 = data["MMSE"].dropna()
    # Our group decided to drop the missing data as to preserve the accuracy of the data
    # Time to handle duplicated values
    
