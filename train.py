import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import preprocessing 
from sklearn import metrics

#####import data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#####Show data statistics
print(train.describe())
print(test.describe())

##print(train.columns[train.isnull().any()].tolist())

print(train.groupby('MSZoning', as_index=False)['YearBuilt'].mean())
print(train.groupby('MSZoning', as_index=False)['Neighborhood'].apply(lambda x: x.value_counts().head(1)))
print(train.groupby('Neighborhood', as_index=False)['YearBuilt'].mean())

#####Fill empty cells with nan

data = [train,test]
numeric_list = ['Alley','Street','PoolQC','MiscFeature','Pool']
encode_list = ['GarageQual','GarageCond','GarageFinish','GarageType','Heating','HeatingQC','CentralAir','ExterCond','ExterQual','MSZoning','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Foundation','PavedDrive','Functional','Electrical','SaleType','SaleCondition','Fence','FireplaceQu','KitchenQual','LotConfig','LandSlope','Neighborhood','LandContour','LotShape','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType']
drop_list = ['Utilities','PoolArea','OpenPorchSF','ScreenPorch']
garage_list = ['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond']
bmst_list = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']

####feature engineering; fill missing values

for i in data:
	k = 0
	for j in i['Id']:
		if i.loc[k,'Alley'] == 'Grvl':
			i.at[k,'Alley'] = 1
		elif i.loc[k,'Alley'] == 'Pave':
			i.at[k,'Alley'] = 2
		else:
			i.at[k,'Alley'] = 0
		if i.loc[k,'Street'] == 'Grvl':
			i.at[k,'Street'] = 1
		elif i.loc[k,'Street'] == 'Pave':
			i.at[k,'Street'] = 2
		if i.loc[k,'PoolQC'] == 'Ex':
			i.at[k,'PoolQC'] = 1
		elif i.loc[k,'PoolQC'] == 'Gd':
			i.at[k,'PoolQC'] = 2
		elif i.loc[k,'PoolQC'] == 'Fa':
			i.at[k,'PoolQC'] = 3
		else:
			i.at[k,'PoolQC'] = 0
		if i.loc[k,'PoolArea'] > 0:
			i.at[k,'Pool'] = 1
		else:
			i.at[k,'Pool'] = 0
		if pd.isna(i.at[k, 'MiscFeature']):
			i.at[k, 'MiscFeature'] = 0
		else:
			i.at[k, 'MiscFeature'] = 1
		if pd.isna(i.at[k, 'Fence']):
			i.at[k, 'Fence'] = 'Abs'
		if pd.isna(i.at[k, 'FireplaceQu']):
			i.at[k, 'FireplaceQu'] = 'Abs'
		if pd.isna(i.at[k, 'MasVnrType']):
			i.at[k, 'MasVnrType'] = 'Abs'
		if pd.isna(i.at[k, 'MasVnrArea']):
			i.at[k, 'MasVnrArea'] = i.loc[i['MasVnrArea'] > 0, 'MasVnrArea'].mean()
		if pd.isna(i.at[k, 'LotFrontage']):
			i.at[k, 'LotFrontage'] = i.loc[i['LotFrontage'] > 0, 'LotFrontage'].mean()		
		for h in garage_list:
			if h == 'GarageYrBlt':
				if pd.isna(i.at[k, 'GarageYrBlt']):
					i.at[k, 'GarageYrBlt'] = 0
			else:
				if pd.isna(i.at[k, h]):
					i.at[k, h] = 'Abs'
		if pd.isna(i.at[k, 'GarageArea']):
			i.at[k, 'GarageArea'] = i.loc[i['GarageArea'] > 0, 'GarageArea'].mean()
		if pd.isna(i.at[k, 'GarageCars']):
			i.at[k, 'GarageCars'] = i.loc[i['GarageCars'] > 0, 'GarageCars'].value_counts().idxmax()
		for h in bmst_list:
			if i.loc[k,'Foundation'] == 'Slab':
				i.at[k, h] = 'Abs'
		# if pd.isna(i.at[k, 'MSZoning']):
		# 	print(i.loc[k,'YearBuilt'])
		# 	##print(i.loc[i['YearBuilt'], 'MSZoning'].most_frequent())
		# 	if i.loc[k,'YearBuilt'] < 1930:
		# 		i.at[k, 'MSZoning'] = 'C (all)'
		# 	elif i.loc[k,'YearBuilt'] < 1960:
		# 		i.at[k, 'MSZoning'] = i.loc[i['Neighborhood'], 'MSZoning'].value_counts().idxmax()				
		k+=1
	for j in numeric_list:
		i[j] = pd.to_numeric(i[j])
	label_encoder = preprocessing.LabelEncoder() 
	for j in encode_list:
		imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
		imputer.fit(i[[j]])
		i[[j]]= imputer.transform(i[[j]])
		i[j]= label_encoder.fit_transform(i[j])
	i = i.drop(drop_list, axis=1)
	##i.drop(drop_list,axis=1,inplace=True)

list_empty=train.columns[train.isnull().any()].tolist()
missing = []
for i in list_empty:
	missing.append([i,train[i].isnull().sum()])
print(missing)

list_empty=test.columns[test.isnull().any()].tolist()
missing = []
for i in list_empty:
	missing.append([i,test[i].isnull().sum()])
print(missing)

#######Feature engineering; create bins

##Things to look into:  LotFrontage,LotArea,MasVnrArea,BsmtFinSF1,BsmtFinSF2,BsmtUnfSF,TotalBsmtSF,1stFlrSF,2ndFlrSF,GrLivArea,LowQualFinSF,GarageArea,OpenPorchSF,EnclosedPorch,3SsnPorch,ScreenPorch

hist_list = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea','LowQualFinSF','GarageArea','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch']

for i in hist_list:
	train[[i]].hist()
	path = 'graphs/{}.png'.format(i)
	plt.savefig(path)
	plt.close
openpsf=(-np.inf, 1, 50, 100, 200, 1000)
screenp=(-np.inf, 1, 100, 200, 1000)
##bsmtsf=(-np.inf, 1, 500, 1000, 2000, 10000)
for i in data:
	i['OpenPorchSF_bin'] = pd.cut(x=i['OpenPorchSF'], bins=openpsf, labels=False)
	i['ScreenPorch_bin'] = pd.cut(x=i['ScreenPorch'], bins=screenp, labels=False)
	##i['TotalBsmtSF_bin'] = pd.cut(x=i['TotalBsmtSF'], bins=bsmtsf, labels=False)


train = train.drop(drop_list, axis=1)
test = test.drop(drop_list, axis=1)

trainMatrix = train.corr()

f, ax = plt.subplots(figsize=(57, 55))

sn.heatmap(trainMatrix, annot=True)
plt.savefig('graphs/heatmap.png')
plt.close
x=0

#####define training and test sets
X_train = train.drop("SalePrice", axis=1)
Y_train = train["SalePrice"]


x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=666)

logreg = LogisticRegression(max_iter=2000)
logreg.fit(x_train, y_train)
Y_pred = logreg.predict(x_test)
acc_log = round(logreg.score(x_train, y_train) * 100, 2)
print('Accuracy Linear Regression:',acc_log)

#####fit data Random Forest
#clf_simple=RandomForestClassifier(n_estimators= 500, random_state=666)
#clf_simple.fit(x_train,y_train)
#Y_pred=clf_simple.predict(x_test)
#print("Accuracy Simple Random Forest:",metrics.accuracy_score(y_test, Y_pred))