import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import preprocessing 
from sklearn import metrics
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

import xgboost as xgb
import catboost as cbr

#####import data
train = pd.read_csv("train.csv")
train.name = 'train'
test = pd.read_csv("test.csv")
test.name = 'test'

#####Show data statistics
print('***Initial Analysis***')
print(train.describe())
print(train.info())

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
print('Scatter plot against SalePrice for each feature in the dataset created.')

data = [train,test]

###Identify and count missing values
for i in data:
	if i.name == 'train':
		mis_train = i.isnull().sum()
		print('Columns with missing values in train:')
		print(mis_train[mis_train > 0])
	elif i.name == 'test':
		mis_test = i.isnull().sum()
		print('Columns with missing values in test:')
		print(mis_test[mis_test > 0])

mis_overall = np.add(mis_train,mis_test)
print('Columns with missing values in total:')
print(mis_overall[mis_overall > 0])

####combine test and train data for analysis
combine_data = pd.concat([train, test], axis=0) 

####analyse correlations for features with missing values
print(combine_data.groupby('MSZoning', as_index=False)['YearBuilt'].mean())
print(combine_data.groupby('MSZoning', as_index=False)['Neighborhood'].apply(lambda x: x.value_counts().head(1)))
print(combine_data.groupby('MasVnrType', as_index=False)['OverallQual'].mean())


numeric_list = ['Alley','Street','MiscFeature','Pool','2ndFlr']
drop_list_columns = ['YearBuilt','YearRemodAdd','1stFlrSF','2ndFlrSF','YrSold','MoSold','BsmtFinSF1','BsmtFinSF2', 'Utilities','Id']  ###Year columns engineered, FlrSF combined and discrete variable for 2nd floor, Time sold+Utilities+Id no correlation, BsmtFin engineered
continuous_list = ['SalePrice','LotArea','LotFrontage','Age','GrLivArea','TotalBsmtSF','GrLivArea','MasVnrArea','BsmtUnfSF','BsmtFinSF1','BsmtFinSF2','1stFlrSF','2ndFlrSF','GarageArea','PorchSF','QualitySum','OverallQual','SinceRenov']
garage_list = ['GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond','GarageArea']
quality_sum_list =['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC']
bath_list = ['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath']
bmst_list = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtUnfSF','BsmtFinSF1']
porch_list = ['OpenPorchSF','ScreenPorch','EnclosedPorch','3SsnPorch','WoodDeckSF']

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
		if pd.isna(i.at[k, 'PoolQC']):
			i.at[k,'PoolQC'] = 'Abs'
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
		if pd.isna(i.at[k, 'MasVnrType']): ######Feature depends on OverallQual; use it as criterion to fill missing values
			if i.at[k, 'OverallQual'] == 7:
				i.at[k, 'MasVnrType'] = 'BrkFace'
			elif i.at[k, 'OverallQual'] > 7:
				i.at[k, 'MasVnrType'] = 'Stone'
			else:
				i.at[k, 'MasVnrType'] = 'None'
		if pd.isna(i.at[k, 'MasVnrArea']): ######no strong correlation with any other feature, fill with data mean
			if i.at[k, 'MasVnrType'] != 'None':
				i.at[k, 'MasVnrArea'] = combine_data.loc[combine_data['MasVnrArea'] > 0, 'MasVnrArea'].mean()
			else:
				i.at[k, 'MasVnrArea'] = 0
		if pd.isna(i.at[k, 'LotFrontage']): ######no strong correlation with any other feature, fill with data mean
			i.at[k, 'LotFrontage'] = combine_data.loc[combine_data['LotFrontage'] > 0, 'LotFrontage'].mean()		
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
		if pd.isna(i.at[k, 'LotFrontage']):  
			i.at[k, 'LotFrontage'] = np.sqrt(com.at[k,"LotArea"]) * combine_data.at["LotFrontage"].mean() / np.sqrt(combine_data.at["LotArea"].mean())
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
		i.at[k, 'QualitySum'] = 0
		for l in quality_sum_list:
			if i.at[k, l] == 'Ex':
				i.at[k, 'QualitySum'] += 5
			elif i.at[k, l] == 'Gd':
				i.at[k, 'QualitySum'] += 4
			elif i.at[k, l] == 'TA':
				i.at[k, 'QualitySum'] += 3
			elif i.at[k, l] == 'Fa':
				i.at[k, 'QualitySum'] += 2
			elif i.at[k, l] == 'Po':
				i.at[k, 'QualitySum'] += 1
		if i.at[k, 'TotalBsmtSF'] > 0:
			i.at[k, 'BsmtUnfSF'] =  i.at[k, 'BsmtUnfSF']/i.at[k, 'TotalBsmtSF']
		i.at[k, 'SinceRenov'] = i.at[k, 'YrSold'] - i.at[k, 'YearRemodAdd']
		if i.at[k, 'SinceRenov'] < 0:
			i.at[k, 'SinceRenov'] = 0
		i.at[k, 'Age'] = i.at[k, 'YrSold'] - i.at[k, 'YearBuilt']
		if i.at[k, 'Age'] < 0:
			i.at[k, 'Age'] = 0
		i.at[k, 'PorchSF'] = i.at[k, 'OpenPorchSF'] + i.at[k, 'EnclosedPorch'] + i.at[k, '3SsnPorch'] + i.at[k, 'ScreenPorch'] + i.at[k, 'WoodDeckSF']
		for h in porch_list:
			if i.at[k, h] > 0:
				i.at[k, h] = 1
			else:
				i.at[k, h] = 0
		i.at[k, 'Bath_count'] = i.at[k, 'BsmtFullBath'] + i.at[k, 'FullBath'] + 0.5*i.at[k, 'BsmtHalfBath'] + 0.5*i.at[k, 'HalfBath']
		i.at[k, 'LotFrontage'] = i.at[k, 'LotFrontage'] / np.sqrt(i.at[k,"LotArea"])
		if pd.isna(i.at[k, 'MSZoning']): #####fill missing values based on 
			i.at[k, 'MSZoning'] = 'C (all)'
			if i.loc[k,'YearBuilt'] > 1938:
				i.at[k, 'MSZoning'] = 'RM'
			if i.loc[k,'YearBuilt'] > 1948:
				i.at[k, 'MSZoning'] = 'RH'
			if i.loc[k,'YearBuilt'] > 1970:
				i.at[k, 'MSZoning'] = 'RL'
			if i.loc[k,'YearBuilt'] > 2000:
				i.at[k, 'MSZoning'] = 'FV'
		k+=1
	for j in numeric_list:
		i[j] = pd.to_numeric(i[j])
	label_encoder = preprocessing.LabelEncoder()
	if i.name == 'train':
		mis_train = i.isnull().sum()
		print('Columns with missing values in train after manual filling:')
		print(mis_train[mis_train > 0])
	if i.name == 'test':
		mis_test = i.isnull().sum()
		print('Columns with missing values in test after manual filling::')
		print(mis_test[mis_test > 0])
	for j in i.columns: ####fill missing values in left over columnt with mos frequent value
		imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
		imputer.fit(i[[j]])
		i[[j]]= imputer.transform(i[[j]])
		if i.dtypes[j] == 'object': #####label encode all columns with str
			i[j]= label_encoder.fit_transform(i[j])

list_empty=train.columns[train.isnull().any()].tolist()
missing = []
for i in list_empty:
	missing.append([i,train[i].isnull().sum()])
if missing:
	print('Some values in the training set are still missing!')
	print(missing)
else:
	print('No missing values in the training set left!')

list_empty=test.columns[test.isnull().any()].tolist()
missing = []
for i in list_empty:
	missing.append([i,test[i].isnull().sum()])
if missing:
	print('Some values in the test set are still missing!')
	print(missing)
else:
	print('No missing values in the test set left!')

####Correlation Matrix
trainMatrix = train.corr()
f, ax = plt.subplots(figsize=(57, 55))
sn.heatmap(trainMatrix, annot=True)
plt.savefig('graphs/heatmap_before.png')
plt.close

#####Histogramms numerical
##hist_list = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea','LowQualFinSF','GarageArea','PorchSF','Bath_count']
f, ax = plt.subplots(4, 5, figsize=(20, 20))
l=1
for i in continuous_list:
	ax = plt.subplot(4,5,l)
	sn.histplot(data=train[i])
	ax.set_title(i)
	l+=1
f.tight_layout()
plt.savefig('graphs/hist_num.png')
plt.close

###transform numerical to obtain normal distribution
#quantile = QuantileTransformer(n_quantiles=1000,output_distribution='normal')
transform_list=['SalePrice','LotArea','LotFrontage','GrLivArea','TotalBsmtSF','GrLivArea','1stFlrSF','GarageArea','PorchSF']
for i in data:
	for col in transform_list:
		if i.name == 'test' and col == 'SalePrice':
			pass
		elif i.name == 'train' and col == 'SalePrice':
			i[col] = np.log1p(i.loc[:,col].values.reshape(-1, 1))
		else:
			#i[col] = quantile.fit_transform(i.loc[:,col].values.reshape(-1, 1))
			i[col] = np.log1p(i.loc[:,col].values.reshape(-1, 1))

f, ax = plt.subplots(4, 5, figsize=(20, 20))
l=1
for i in continuous_list:
	ax = plt.subplot(4,5,l)
	sn.histplot(data=train[i])
	ax.set_title(i)
	l+=1
f.tight_layout()
plt.savefig('graphs/hist_num_quantile.png')
plt.close

f, ax = plt.subplots(4, 5, figsize=(20, 20))
l=1
for i in continuous_list:
	ax = plt.subplot(4,5,l)
	sn.boxplot(data=train[i])
	ax.set_title(i)
	l+=1
plt.savefig('graphs/box_num_quantile.png')
plt.close


####delete defined columns
for i in data:
	for col in drop_list_columns:
		i.drop(col, axis=1, inplace=True)
	for col in bath_list:
		i.drop(col, axis=1, inplace=True)
	for col in garage_list:
		i.drop(col, axis=1, inplace=True)
	i = i.rename({'OpenPorchSF': 'OpenPorch', 'WoodDeckSF': 'WoodDeck'}, axis=1)
	for col in quality_sum_list:
		if (col == 'GarageQual' or col == 'GarageCond'):
			pass
		else:
			i.drop(col, axis=1, inplace=True)

###remove columns with low correlation to price

# print('\n***Remove features with low correlation to SalePrice and analyse correlations between remaining features***')
# drop_cutoff = []
# for col in train.columns:
# 	if np.abs(train['SalePrice'].corr(train[col])) < 0.1:
# 		drop_cutoff.append([col, train['SalePrice'].corr(train[col])])
# 		train.drop(col, axis=1, inplace=True)
# 		test.drop(col, axis=1, inplace=True)
# print('The following features were dropped due to low correlation:')
# print(*drop_cutoff, sep = '\n')

###check for high correlation between features, remove worse feature when correlation higher than 0.8
high_corr = []
drop_corr = []
col1_list = []
col2_list = []
for col1 in train.columns:
	for col2 in train.columns:
		col2_list.append(col2)
		if col1 == col2 or col1 == 'SalePrice' or col2 == 'SalePrice':
			pass
		else:
			if ((col1 in col2_list) & (col2 in col1_list)):
				pass
			else:
				if np.abs(train[col1].corr(train[col2])) > 0.6:
					high_corr.append([col1, col2, train[col1].corr(train[col2])])
				if np.abs(train[col1].corr(train[col2])) > 0.9 and np.abs(train['SalePrice'].corr(train[col1])) > np.abs(train['SalePrice'].corr(train[col2])):
					if col2 not in drop_corr:
						drop_corr.append(col2)
				elif np.abs(train[col1].corr(train[col2])) > 0.9 and np.abs(train['SalePrice'].corr(train[col2])) > np.abs(train['SalePrice'].corr(train[col1])):
					if col1 not in drop_corr:
						drop_corr.append(col1)
		col2_list.append(col2)
	col1_list.append(col1)

for col in drop_corr:
	train.drop(col, axis=1, inplace=True)
	test.drop(col, axis=1, inplace=True)

print('\nThe follwoing features show a high correlation! Check whether one of them can  be removed.')
print(*high_corr, sep = '\n')

print('\nAutomatically removed:')
print(*drop_corr, sep = '\n')

###check skewness of transformed features
cont_skew = []
for col in transform_list:
	try:
		cont_skew.append([col, train[col].skew()])
	except:
		pass
print('\nSkewness of continuous features:')
print(*cont_skew, sep = '\n')

###check for outliers
cont_outlier = []
for col in transform_list:
	try:
		Q1, Q3 = np.percentile(train[col], 25), np.percentile(train[col], 75)
		IQR = Q3 - Q1
		cut_off = IQR * 3
		lower = Q1 - cut_off 
		upper = Q3 + cut_off
		outliers = [x for x in train[col] if x < lower or x > upper]
		cont_outlier.append([col, len(outliers)])
	except:
		pass
print('\nOutliers in continuous features (3*IQR):')
print(*cont_outlier, sep = '\n')

###plot new distributions
f, ax = plt.subplots(8, 9, figsize=(40, 40))
l=1
for col in train.columns:
	ax = plt.subplot(8,9,l)
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

print('\n***Performance of various models***')
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
print("Accuracy Random Forest (RMSLE):",np.sqrt(metrics.mean_squared_log_error(np.expm1(y_test), np.expm1(Y_pred))))

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
f,ax = plt.subplots(figsize=(10,10))
xgb.plot_importance(xg_reg,ax=ax,color='red')
plt.savefig('graphs/feature_contribution_xgboost.png')
plt.close
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
# xg_reg_opt = GridSearchCV(estimator = xgb.XGBRegressor(), param_grid = param_grid, scoring = 'neg_mean_squared_error', cv = 5, verbose = 1)
# xg_reg_opt.fit(x_train, y_train)
# print("Best parameters XGBoost: ",xg_reg_opt.best_params_)
# Y_pred = xg_reg_opt.predict(x_test)
# print("Accuracy Optimized XGBoost (RMSLE):",np.sqrt(metrics.mean_squared_log_error(np.expm1(y_test), np.expm1(Y_pred))))
## Best parameters XGBoost:  {'colsample_bytree': 0.475, 'learning_rate': 0.05, 'max_depth': 40, 'n_estimators': 200}
## Best parameters XGBoost:  {'booster': 'gbtree', 'colsample_bytree': 0.3, 'learning_rate': 0.05, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 1000, 'objective': 'reg:squarederror'}

####CatBoost
cb_reg = cbr.CatBoostRegressor(n_estimators = 1000, loss_function = 'MAE', eval_metric = 'MSLE')
cb_reg.fit(x_train,y_train,silent=True)
Y_pred = cb_reg.predict(x_test)
##print("Accuracy XGBoost (RMSLE):",np.sqrt(metrics.mean_squared_log_error(y_test, Y_pred)))
print("Accuracy CatBoost (RMSLE):",np.sqrt(metrics.mean_squared_log_error(np.expm1(y_test), np.expm1(Y_pred))))