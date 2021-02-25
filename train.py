import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import seaborn as sn
import math

from collections import Counter

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import preprocessing 
from sklearn import metrics
from sklearn.preprocessing import QuantileTransformer


#####import data
train = pd.read_csv("train.csv")
train.name = 'train'
test = pd.read_csv("test.csv")
test.name = 'test'

#####Show data statistics
print(train.describe())
print(test.describe())

f, ax = plt.subplots(9, 9, figsize=(50, 50))
l=1
for col in train.columns:
	if not col == 'SalePrice':
		ax = plt.subplot(9,9,l)
		sn.scatterplot(x=col,y='SalePrice',data=train)
		ax.set_title(col)
		l+=1
f.tight_layout()
plt.savefig('graphs/scatter_price_all.png')
plt.close


print(train.groupby('MSZoning', as_index=False)['YearBuilt'].mean())
print(train.groupby('MSZoning', as_index=False)['Neighborhood'].apply(lambda x: x.value_counts().head(1)))
print(train.groupby('Neighborhood', as_index=False)['YearBuilt'].mean())

data = [train,test]
numeric_list = ['Alley','Street','PoolQC','MiscFeature','Pool','2ndFlr']
encode_list = ['GarageQual','GarageCond','GarageFinish','GarageType','Heating','HeatingQC','CentralAir','ExterCond','ExterQual','MSZoning','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Foundation','PavedDrive','Functional','Electrical','SaleType','SaleCondition','Fence','FireplaceQu','KitchenQual','LotConfig','LandSlope','Neighborhood','LandContour','LotShape','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType']
drop_list_columns = ['Utilities','PoolArea','FireplaceQu','MasVnrType','TotRmsAbvGrd','Condition2','LandContour','Id','MiscVal','YrSold','LowQualFinSF','BsmtFinType1','BsmtFinType2','PoolQC','Street','MoSold','LandSlope','Exterior2nd','MSSubClass','Foundation','SaleType','LotConfig','2ndBsmtFlr','1stFlrSF','2ndFlrSF']
quantile_list = ['SalePrice','LotArea','LotFrontage','YearBuilt','GrLivArea','TotalBsmtSF','GrLivArea','MasVnrArea','BsmtUnfSF','BsmtFinSF1','BsmtFinSF2','1stFlrSF','2ndFlrSF','GarageArea','WoodDeckSF','PorchSF']
garage_list = ['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','GarageArea']
bath_list = ['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath']
bmst_list = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtUnfSF','BsmtFinSF1']
porch_list = ['OpenPorchSF','ScreenPorch','EnclosedPorch','3SsnPorch']

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
				if pd.isna(i.at[k, h]) and not h == 'GarageArea':
					i.at[k, h] = 'Abs'
		if pd.isna(i.at[k, 'GarageArea']):
			i.at[k, 'GarageArea'] = i.loc[i['GarageArea'] > 0, 'GarageArea'].mean()
		if pd.isna(i.at[k, 'GarageCars']):
			i.at[k, 'GarageCars'] = i.loc[i['GarageCars'] > 0, 'GarageCars'].value_counts().idxmax()
		if pd.isna(i.at[k, 'TotalBsmtSF']):
			i.at[k, 'TotalBsmtSF'] = i.at[k, '1stFlrSF']
		for h in bath_list:
			if pd.isna(i.at[k, h]):
				if i.at[k, 'TotalBsmtSF'] > 0 and h == 'BsmtFullBath':
					i.at[k, 'BsmtFullBath'] = i.loc[i['TotalBsmtSF'] > 0, 'BsmtFullBath'].value_counts().idxmax()
				elif i.at[k, 'TotalBsmtSF'] == 0 and h == 'BsmtFullBath':
					i.at[k, 'BsmtFullBath'] = 0
				elif i.at[k, 'TotalBsmtSF'] > 0 and h == 'BsmtHalfBath':
					i.at[k, 'BsmtHalfBath'] = i.loc[i['TotalBsmtSF'] > 0, 'BsmtHalfBath'].value_counts().idxmax()
				elif i.at[k, 'TotalBsmtSF'] == 0 and h == 'BsmtHalfBath':
					i.at[k, 'BsmtHalfBath'] = 0
		for h in porch_list:
			if i.at[k, h] > 0:
				i.at[k, h] = 1
			else:
				i.at[k, h] = 0
		for h in bmst_list:
			if i.loc[k,'Foundation'] == 'Slab' and not (h == 'BsmtUnfSF' or h == 'BsmtFinSF1'):
				i.at[k, h] = 'Abs'
		if i.at[k, '2ndFlrSF'] > 0:
			i.at[k, '2ndFlr'] = 1
		else:
			i.at[k, '2ndFlr'] = 0
		if i.at[k, 'BsmtFinSF2'] > 0:
			i.at[k, '2ndBsmtFlr'] = 1
		else:
			i.at[k, '2ndBsmtFlr'] = 0
		##i.at[k, 'BuildSF'] = i.at[k, '1stFlrSFSF'] + i.at[k, '2ndFlrSF']
		i.at[k, 'PorchSF'] = i.at[k, 'OpenPorchSF'] + i.at[k, 'EnclosedPorch'] + i.at[k, '3SsnPorch'] + i.at[k, 'ScreenPorch']
		i.at[k, 'Bath_count'] = i.at[k, 'BsmtFullBath'] + i.at[k, 'FullBath'] + 0.5*i.at[k, 'BsmtHalfBath'] + 0.5*i.at[k, 'HalfBath']
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

####Correlation Matrix
trainMatrix = train.corr()
f, ax = plt.subplots(figsize=(57, 55))
sn.heatmap(trainMatrix, annot=True)
plt.savefig('graphs/heatmap_before.png')
plt.close

#####Histogramms numerical
hist_list = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea','LowQualFinSF','GarageArea','PorchSF','Bath_count']
f, ax = plt.subplots(4, 4, figsize=(20, 20))
l=1
for i in hist_list:
	ax = plt.subplot(4,4,l)
	sn.histplot(data=train[i])
	ax.set_title(i)
	l+=1
f.tight_layout()
plt.savefig('graphs/hist_num.png')
plt.close

###transform numerical to obtain normal distribution
quantile = QuantileTransformer(n_quantiles=1000,output_distribution='normal')
for i in data:
	for col in quantile_list:
		if i.name == 'test' and col == 'SalePrice':
			pass
		elif i.name == 'train' and col == 'SalePrice':
			i[col] = np.log1p(i.loc[:,col].values.reshape(-1, 1))
		else:
			i[col] = quantile.fit_transform(i.loc[:,col].values.reshape(-1, 1))

####delete defined columns
for i in data:
	for col in drop_list_columns:
		i.drop(col, axis=1, inplace=True)

for i in data:
	for col in bath_list:
		i.drop(col, axis=1, inplace=True)

for i in data:
	for col in garage_list:
		i.drop(col, axis=1, inplace=True)

###remove columns with low correlation to price
for col in train.columns:
	if math.sqrt((train['SalePrice'].corr(train[col]))**2) < 0.2:
		train.drop(col, axis=1, inplace=True)
		test.drop(col, axis=1, inplace=True)

###plot new distributions
f, ax = plt.subplots(8, 7, figsize=(40, 40))
l=1
for col in train.columns:
	ax = plt.subplot(8,7,l)
	sn.histplot(data=train[col],kde=True)
	ax.set_title(col)
	l+=1

f.tight_layout()
plt.savefig('graphs/hist_after_distr.png')
plt.close

###final correlation matrix
trainMatrix = train.corr()
f, ax = plt.subplots(figsize=(32, 30))
sn.heatmap(trainMatrix, annot=True)
plt.savefig('graphs/heatmap_cutoff.png')
plt.close

##print(train.corr()['SalePrice'].sort_values())

#####define training and test sets
X_train = train.drop("SalePrice", axis=1)
Y_train = train["SalePrice"]
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=666)

######LinReg

linreg = LinearRegression()
linreg.fit(x_train, y_train)
Y_pred = linreg.predict(x_test)
##print("Accuracy Linear Regression (RMSLE):",np.sqrt(metrics.mean_squared_log_error(y_test, Y_pred)))
print("Accuracy Linear Regression (RMSLE):",np.sqrt(metrics.mean_squared_log_error(np.expm1(y_test), np.expm1(Y_pred))))
#####Random Forest
clf_simple=RandomForestRegressor(n_estimators= 500, random_state=666)
clf_simple.fit(x_train,y_train)
Y_pred=clf_simple.predict(x_test)
##print("Accuracy Simple Random Forest (RMSLE):",np.sqrt(metrics.mean_squared_log_error(y_test, Y_pred)))
print("Accuracy Simple Random Forest (RMSLE):",np.sqrt(metrics.mean_squared_log_error(np.expm1(y_test), np.expm1(Y_pred))))

#####Optimize Random Forest

# clf=RandomForestRegressor(random_state=666)
# param_grid = { 
#     'n_estimators': [200],
#     'min_samples_split': [2, 5, 10, 15],
# 	'min_samples_leaf': [3, 5, 8, 12],
#     'max_features': ['auto', 'sqrt', 'log2', None],
#     'max_depth' : [5, 10, 20, 30],
#     'criterion' :['gini', 'entropy'] 
# }
# CV_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5, scoring = 'neg_mean_squared_error', verbose = 1)
# CV_clf.fit(x_train, y_train)
# print('Optimized Random Forest Classifier:',CV_clf.best_params_)
# Y_pred=CV_clf.predict(x_test)
# print("Accuracy Optimized Random Forest (RMSLE):",np.sqrt(metrics.mean_squared_log_error(np.expm1(y_test), np.expm1(Y_pred))))
## Optimized Random Forest Classifier: {'criterion': 'gini', 'max_depth': 20, 'max_features': 'log2', 'min_samples_leaf': 3, 'min_samples_split': 10, 'n_estimators': 200}

####XGBoost
xg_reg = xgb.XGBRegressor(objective = 'reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.05,max_depth = 5, n_estimators = 1000, booster = 'gbtree')
xg_reg.fit(x_train,y_train)
Y_pred = xg_reg.predict(x_test)
##print("Accuracy XGBoost (RMSLE):",np.sqrt(metrics.mean_squared_log_error(y_test, Y_pred)))
print("Accuracy XGBoost (RMSLE):",np.sqrt(metrics.mean_squared_log_error(np.expm1(y_test), np.expm1(Y_pred))))

####Optimized XGBoost
# param_grid = {
#      'colsample_bytree': np.linspace(0.3, 1, 4),
#      'n_estimators':[100,500,1000],
#      'max_depth': [3, 5, 7, 10],
#      'learning_rate': [0.05, 0.1,0.15],
#      'min_child_weight': [1,3,6],
#      'booster': ['gbtree'],
#      'objective': ['reg:squarederror']
# }
# xg_reg_opt = GridSearchCV(estimator = xgb.XGBRegressor(), param_grid = param_grid, scoring = 'neg_mean_squared_error', cv = 3, verbose = 1)
# xg_reg_opt.fit(x_train, y_train)
# print("Best parameters XGBoost: ",xg_reg_opt.best_params_)
# Y_pred = xg_reg_opt.predict(x_test)
# print("Accuracy Optimized XGBoost (RMSLE):",np.sqrt(metrics.mean_squared_log_error(np.expm1(y_test), np.expm1(Y_pred))))
## Best parameters XGBoost:  {'colsample_bytree': 0.475, 'learning_rate': 0.05, 'max_depth': 40, 'n_estimators': 200}
