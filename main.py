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

#Since most of this data is missing and since such data does not seem to be if high correlation with our dependent variable. let's go ahead and drop them!

# columns to be dropped
columnsToDrop = missingData[missingData['Percentage']>50].index

all_data = all_data.drop(columnsToDrop, axis=1)
# test = test.drop(columnsToDrop, axis=1)
print(all_data.shape)

"""
Handling categorcial missing data

We will replace missing data for the catigorical features with None
BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : For all these categorical basement-related features, NaN means that there is no basement.
FireplaceQu, GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None
"""

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 
            'BsmtFinType1', 'BsmtFinType2','BsmtFullBath', 'BsmtHalfBath',
            'GarageType', 'GarageFinish', 'GarageQual', 'BsmtUnfSF','BsmtFinSF1','BsmtFinSF2',
            'GarageCond', 'FireplaceQu', 'MasVnrType', 'Exterior2nd'):
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna('None')

# Handling numerical missing data

#GarageYrBlt replacing missing data with 0
all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(0)

# NA most likely means no masonry veneer for these houses. We can fill in 0
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

# let's drop YrSold since it's also not correlated with 'SalePrice'
all_data = all_data.drop('YrSold', axis=1)

# Electrical has one NA value. Since this feature has mostly 'SBrkr', we can set that for the missing value.
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

# 'RL' is by far the most common value. So we can fill in missing values with 'RL'
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

# For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . 
# Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling.
all_data = all_data.drop(['Utilities'], axis=1)

# data description says NA means typical
all_data["Functional"] = all_data["Functional"].fillna("Typ")

# Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))


#  Replacing missing data with 0 (Since missing in this case would imply 0.)
for col in ('TotalBsmtSF', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
    
#  Replacing missing data with the most common
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])


"""
Stage 2: Outliers!

In statistics, an outlier is an observation point that is distant from other observations. usually the distance is measured by standard deviations. such points are usually produced by some sort of error or simply do not represent any real data and just get in the way to make our predictions less accurate.

The approach we're going to go with is simply remove data that's below the 0.05 percentile or above the 0.9 percentile (check out this link to better understand quantiles and percentiles: http://www.statisticshowto.com/quantile-definition-find-easy-steps/).
"""

from pandas.api.types import is_numeric_dtype
def remove_outliers(df):
    low = .05
    high = .9
    quant_df = df.quantile([low, high])
    for name in list(df.columns):
        if is_numeric_dtype(df[name]):
            df = df[(df[name] > quant_df.loc[low, name]) & (df[name] < quant_df.loc[high, name])]
    return df

remove_outliers(all_data).head()

# doesn't see like we have any outliers in the chosen quantile.
# Nonetheless, i would like to explore the feature 'GrLivArea' and see if i could visually spot outliers.

plt.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

# We can see at the bottom right two with extremely large GrLivArea that are of a low price. These values are huge oultliers (those bastards). Therefore, we can safely delete them
# Deleting outliers
tempTrain = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

plt.scatter(x = tempTrain['GrLivArea'], y = tempTrain['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()



