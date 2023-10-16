import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# importing some libraries for visulizations
import matplotlib.pyplot as plt
import seaborn as sns

# importing sklearn to select the model that will be fitting out data into
# we will train_test_split to divide the data
# we will use cross_val_score to determine best accuracy 
from sklearn.model_selection import train_test_split, cross_val_score

# import the data into dataframes using pandas library
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head()

cols = {}
uniqueCols =[]
for col in test.columns:
    if col not in cols:
        cols[col]=1
    else:
        cols+=1
for col in train.columns:
    if col not in cols:
        uniqueCols.append(col)


#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']


# drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

# we will look also at it's skewness (if it's equal to 0 it means that this variable is evenly distributed) and kurtosis (the standard value should be 3)
sns.distplot(train['SalePrice'], bins=20, rug=True)

print("Skewness: %0.2f" %train['SalePrice'].skew())
print("Kurtosis: %0.2f" %train['SalePrice'].kurt())

# good practice (and also the best first move) when doing DS projects is to look for all sorts of correlations between all features.
corrmat = train.corr()
plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=.8, annot=True);

# When we have TOO MANY features, it is best to filter correlations first. in this case we will settle for 0.5 correlation or above.

corrmat = train.corr()
# extracting the relevant features
filteredCorrMat_features = corrmat.index[abs(corrmat['SalePrice'])>=0.5]
plt.figure(figsize=(12,12))
# performing corr on the chosen features and presenting it on the heatmap
sns.heatmap(train[filteredCorrMat_features].corr(),annot=True,cmap='winter')

"""
In this way, we selected only the most important features that will serve us as the best predictors for SalePrice.

Furthermore, we find that the columns 'OverallQaul', 'GrLivArea' have the highest corrlations with SalePrice.

It is also very important to notice correlations amongst other features like:

'GrLivArea' and 'TotalRmsAbvGrd' (corr= 0.83)
'GarageCars' and 'GarageArea' (corr= 0.88)
'lstFlrSF' and 'TotalBsmtSF' (corr= 0.82)
It seems like OverallQaul serves as the most reliable feature for predicting SalePrice

"""

sns.barplot(train.OverallQual,train.SalePrice)

# Before we dive into feature engineering, let's join our training data and test data so that we won't get lost later and stay consistent with changes across the data
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))

"""
Pre-processing and Feature Engineering

Stage 1: Handling Missing Data
There are many considerations when we are dealing with missing data. As a first step, let's see which data is missing and it's weight in percentage.

"""

totalMissing = all_data.isnull().sum().sort_values(ascending=False)
percentage = ((all_data.isnull().sum()/all_data.isnull().count())*100).sort_values(ascending=False)

missingData = pd.concat([totalMissing,percentage],axis=1,keys=['Total','Percentage'])
missingData.head(20)

# visual representation
plt.subplots(figsize=(15,20))
plt.xticks(rotation='90')
sns.barplot(x=totalMissing.index[:24],y=percentage[:24])
plt.xlabel('features')
plt.ylabel('percentage of missing data')
plt.title('percent of missing data by feature')
plt.show()



