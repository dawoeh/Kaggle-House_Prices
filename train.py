import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import seaborn as sn

from sklearn.impute import SimpleImputer
from sklearn import preprocessing 
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from skopt import BayesSearchCV
from skopt.space import Real

import xgboost as xgb
import catboost as cbr

#####FUNCTIONS
def correlation_columns(df, target_feature, cor_limit, drop_limit):
	high_corr = []
	drop_corr = []
	col1_list = []
	col2_list = []
	for col1 in df.columns:
		for col2 in df.columns:
			col2_list.append(col2)
			if col1 == col2 or col1 == target_feature or col2 == target_feature:
				pass
			else:
				if ((col1 in col2_list) and (col2 in col1_list)):
					pass
				else:
					if np.abs(df[col1].corr(df[col2])) > cor_limit:
						high_corr.append([col1, col2, df[col1].corr(df[col2])])
					if np.abs(df[col1].corr(df[col2])) > drop_limit and np.abs(df[target_feature].corr(df[col1])) >= np.abs(df[target_feature].corr(df[col2])):
						if col2 not in drop_corr:
							drop_corr.append(col2)
					elif np.abs(df[col1].corr(df[col2])) > drop_limit and np.abs(df[target_feature].corr(df[col2])) > np.abs(df[target_feature].corr(df[col1])):
						if col1 not in drop_corr:
							drop_corr.append(col1)
		col1_list.append(col1)
	return (high_corr, drop_corr)

def print_status(optimal_result):
	models_tested = pd.DataFrame(bayes_cv_tuner.cv_results_)
	best_parameters_so_far = pd.Series(bayes_cv_tuner.best_params_)
	print(
		'Model #{}\nBest so far: {}\nBest parameters so far: {}\n'.format(
			len(models_tested),
			np.round(bayes_cv_tuner.best_score_, 5),
			bayes_cv_tuner.best_params_,
		)
	)
	clf_type = bayes_cv_tuner.estimator.__class__.__name__
	models_tested.to_csv(clf_type + '_cv_results_summary.csv')

def return_binary_col(df):
	binary_list = df.columns[df.isin([0,1]).all()]
	return binary_list

def low_corr_target(df, target_feature, cut_off):
	drop_cutoff = {}
	for col in df.columns:
		if np.abs(df[target_feature].corr(df[col])) < cut_off:
			drop_cutoff[col] = df[target_feature].corr(df[col])
	return drop_cutoff

def count_outliers(df, columns, IQR_range):
	dict_outlier = {}
	for col in [element for element in columns if element in df]:
		Q1, Q3 = np.percentile(df[col], 25), np.percentile(df[col], 75)
		IQR = Q3 - Q1
		cut_off = IQR * IQR_range
		lower = Q1 - cut_off 
		upper = Q3 + cut_off
		outliers = [x for x in df[col] if x < lower or x > upper]
		dict_outlier[col] = len(outliers)
	return dict_outlier

#####DATA IMPORT
train = pd.read_csv('train.csv')
train.name = 'train'
test = pd.read_csv('test.csv')
test.name = 'test'

#####DATA STATISTICS
print('***Initial Analysis***')
print(train.describe())
print(train.info())

#####SCATTER PLOT OF ALL FEATURES
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

#####MISSING DATA
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

#####COMBINE TEST AND TRAIN FOR ANALYSIS
train['Test'] = 0
test['Test'] = 1
combine_data = pd.concat([train, test], axis=0, ignore_index=True)
print(combine_data.describe())

#####CORRELATIONS FOR MISSING DATA
print(combine_data.groupby('MSZoning', as_index=False)['YearBuilt'].mean())
print(combine_data.groupby('MSZoning', as_index=False)['Neighborhood'].apply(lambda x: x.value_counts().head(1)))
print(combine_data.groupby('MasVnrType', as_index=False)['OverallQual'].mean())

continuous_list = ['SalePrice','LotArea','LotFrontage','Age','GrLivArea','TotalBsmtSF','MasVnrArea','BsmtFinSF1','BsmtFinSF2','1stFlrSF','2ndFlrSF','GarageArea','PorchSF','QualitySum','OverallQual','SinceRenov']
garage_list = ['GarageType','GarageFinish','GarageQual','GarageCond']
quality_sum_list =['ExterQual','BsmtQual','HeatingQC','KitchenQual','FireplaceQu','GarageQual','ExterCond', 'BsmtCond', 'GarageCond', 'PoolQC']
bath_list = ['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath']
bmst_list = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtUnfSF','BsmtFinSF1']
porch_list = ['OpenPorchSF','ScreenPorch','EnclosedPorch','3SsnPorch','WoodDeckSF']

#####FEATURE ENGINEERING; FILL MISSING DATA MANUALLY

combine_data.loc[combine_data['Alley'].isna(), 'Alley'] = 'Abs'
combine_data.loc[combine_data['PoolQC'].isna(), 'PoolQC'] = 'Abs'
combine_data.loc[combine_data['Fence'].isna(), 'Fence'] = 'Abs'
combine_data.loc[combine_data['FireplaceQu'].isna(), 'FireplaceQu'] = 'Abs'
combine_data.loc[combine_data['MasVnrType'].isna(), 'MasVnrType'] = 'Abs'
combine_data.loc[combine_data['MiscFeature'].isna(), 'MiscFeature'] = 'Abs'
for col in garage_list:
	combine_data.loc[combine_data[col].isna(), col] = 'Abs'

combine_data.loc[combine_data['TotalBsmtSF'].isna(), 'TotalBsmtSF'] = 0
combine_data.loc[combine_data['LotFrontage'].isna(), 'LotFrontage'] = 0
combine_data.loc[combine_data['GarageYrBlt'].isna(), 'GarageYrBlt'] = 0
combine_data.loc[combine_data['GarageArea'].isna(), 'GarageArea'] = 0
combine_data.loc[combine_data['GarageCars'].isna(), 'GarageCars'] = 0

k = 0
while k < len(combine_data):
	if combine_data.at[k,'PoolArea'] > 0:
		combine_data.at[k,'Pool'] = 1
	else:
		combine_data.at[k,'Pool'] = 0
	# if pd.isna(combine_data.at[k, 'MiscFeature']):
	# 	combine_data.at[k, 'MiscFeature'] = 0
	# else:
	# 	combine_data.at[k, 'MiscFeature'] = 1
	# if pd.isna(combine_data.at[k, 'MasVnrType']): ######Feature depends on OverallQual; use it as criterion to fill missing values
	# 	if combine_data.at[k, 'OverallQual'] == 7:
	# 		combine_data.at[k, 'MasVnrType'] = 'BrkFace'
	# 	elif combine_data.at[k, 'OverallQual'] > 7:
	# 		combine_data.at[k, 'MasVnrType'] = 'Stone'
	# 	else:
	# 		combine_data.at[k, 'MasVnrType'] = 'Abs'
	if pd.isna(combine_data.at[k, 'MasVnrArea']): ######No strong correlation with any other feature, fill with data mean
		if combine_data.at[k, 'MasVnrType'] != 'Abs':
			combine_data.at[k, 'MasVnrArea'] = combine_data.loc[combine_data['MasVnrArea'] > 0, 'MasVnrArea'].mean()
		else:
			combine_data.at[k, 'MasVnrArea'] = 0
	# if pd.isna(combine_data.at[k, 'LotFrontage']): ######No strong correlation with any other feature, fill with data mean
	# 	combine_data.at[k, 'LotFrontage'] = combine_data.loc[combine_data['LotFrontage'] > 0, 'LotFrontage'].mean()		
	# for col in garage_list:
	# 	if pd.isna(combine_data.at[k, col]) and not col == 'GarageArea':
	# 		combine_data.at[k, col] = 'Abs'
	# if pd.isna(combine_data.at[k, 'GarageArea']):
	# 	combine_data.at[k, 'GarageArea'] = combine_data.loc[combine_data['GarageArea'] > 0, 'GarageArea'].mean()
	# if pd.isna(combine_data.at[k, 'GarageCars']):
	# 	combine_data.at[k, 'GarageCars'] = combine_data.loc[combine_data['GarageCars'] > 0, 'GarageCars'].value_counts().idxmax()
	# if pd.isna(combine_data.at[k, 'LotFrontage']):  
	# 	combine_data.at[k, 'LotFrontage'] = np.sqrt(combine_data.at[k,'LotArea']) * combine_data.at['LotFrontage'].mean() / np.sqrt(combine_data.at['LotArea'].mean())
	for col in bath_list:
		if pd.isna(combine_data.at[k, col]):
			if combine_data.at[k, 'TotalBsmtSF'] > 0 and col == 'BsmtFullBath':
				combine_data.at[k, 'BsmtFullBath'] = combine_data.loc[combine_data['TotalBsmtSF'] > 0, 'BsmtFullBath'].value_counts().idxmax()
			elif combine_data.at[k, 'TotalBsmtSF'] == 0 and col == 'BsmtFullBath':
				combine_data.at[k, 'BsmtFullBath'] = 0
			elif combine_data.at[k, 'TotalBsmtSF'] > 0 and col == 'BsmtHalfBath':
				combine_data.at[k, 'BsmtHalfBath'] = combine_data.loc[combine_data['TotalBsmtSF'] > 0, 'BsmtHalfBath'].value_counts().idxmax()
			elif combine_data.at[k, 'TotalBsmtSF'] == 0 and col == 'BsmtHalfBath':
				combine_data.at[k, 'BsmtHalfBath'] = 0
	for col in bmst_list:
		if combine_data.loc[k,'Foundation'] == 'Slab' and not (col == 'BsmtUnfSF' or col == 'BsmtFinSF1'):
			combine_data.at[k, col] = 'Abs'
	if combine_data.at[k, '2ndFlrSF'] > 0:
		combine_data.at[k, '2ndFlr'] = 1
	else:
		combine_data.at[k, '2ndFlr'] = 0
	if combine_data.at[k, 'BsmtFinSF2'] > 0:
		combine_data.at[k, '2ndBsmtFlr'] = 1
	else:
		combine_data.at[k, '2ndBsmtFlr'] = 0
	combine_data.at[k, 'QualitySum'] = 0
	for l in quality_sum_list:
		if combine_data.at[k, l] == 'Ex':
			combine_data.at[k, 'QualitySum'] += 5
		elif combine_data.at[k, l] == 'Gd':
			combine_data.at[k, 'QualitySum'] += 4
		elif combine_data.at[k, l] == 'TA':
			combine_data.at[k, 'QualitySum'] += 3
		elif combine_data.at[k, l] == 'Fa':
			combine_data.at[k, 'QualitySum'] += 2
		elif combine_data.at[k, l] == 'Po':
			combine_data.at[k, 'QualitySum'] += 1
	if combine_data.at[k, 'TotalBsmtSF'] > 0:
		combine_data.at[k, 'BsmtUnfSF'] =  combine_data.at[k, 'BsmtUnfSF']/combine_data.at[k, 'TotalBsmtSF']
	if combine_data.at[k, 'TotalBsmtSF'] == 0:
		combine_data.at[k, 'BsmtFinType1'] = 'Abs'
		combine_data.at[k, 'BsmtFinType2'] = 'Abs'
		combine_data.at[k, 'BsmtExposure'] = 'Abs'
		combine_data.at[k, 'BsmtQual'] = 'Abs'
		combine_data.at[k, 'BsmtCond'] = 'Abs'
	combine_data.at[k, 'SinceRenov'] = combine_data.at[k, 'YrSold'] - combine_data.at[k, 'YearRemodAdd']
	if combine_data.at[k, 'SinceRenov'] < 0:
		combine_data.at[k, 'SinceRenov'] = 0
	combine_data.at[k, 'Age'] = combine_data.at[k, 'YrSold'] - combine_data.at[k, 'YearBuilt']
	if combine_data.at[k, 'Age'] < 0:
		combine_data.at[k, 'Age'] = 0
	combine_data.at[k, 'PorchSF'] = combine_data.at[k, 'OpenPorchSF'] + combine_data.at[k, 'EnclosedPorch'] + combine_data.at[k, '3SsnPorch'] + combine_data.at[k, 'ScreenPorch'] + combine_data.at[k, 'WoodDeckSF']
	# for h in porch_list:
	# 	if combine_data.at[k, h] > 0:
	# 		combine_data.at[k, h] = 1
	# 	else:
	# 		combine_data.at[k, h] = 0
	combine_data.at[k, 'Bath_count'] = combine_data.at[k, 'BsmtFullBath'] + combine_data.at[k, 'FullBath'] + 0.5*combine_data.at[k, 'BsmtHalfBath'] + 0.5*combine_data.at[k, 'HalfBath']
	combine_data.at[k, 'LotFrontage'] = combine_data.at[k, 'LotFrontage'] / np.sqrt(combine_data.at[k,'LotArea'])
	if pd.isna(combine_data.at[k, 'MSZoning']): #####Fill missing values based on 
		combine_data.at[k, 'MSZoning'] = 'C (all)'
		if combine_data.at[k,'YearBuilt'] > 1938:
			combine_data.at[k, 'MSZoning'] = 'RM'
		if combine_data.at[k,'YearBuilt'] > 1948:
			combine_data.at[k, 'MSZoning'] = 'RH'
		if combine_data.at[k,'YearBuilt'] > 1970:
			combine_data.at[k, 'MSZoning'] = 'RL'
		if combine_data.at[k,'YearBuilt'] > 2000:
			combine_data.at[k, 'MSZoning'] = 'FV'
	k+=1
# for col in numeric_list:
# 	combine_data[col] = pd.to_numeric(combine_data[col])

mis_train = combine_data.isnull().sum()
print('Columns with missing values after manual filling:')
print(mis_train[mis_train > 0])		
for col in combine_data.columns: ####Fill missing values in left over columnt with most frequent value
	if col not in 'SalePrice':
		imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
		imputer.fit(combine_data[[col]])
		combine_data[[col]] = imputer.transform(combine_data[[col]])

list_empty=combine_data.columns[combine_data.isnull().any()].tolist()
missing = []
for col in list_empty:
	missing.append([col,combine_data[col].isnull().sum()])
if missing:
	print('Some values in the data sets are still missing!')
	print(missing)
else:
	print('No missing values in the data sets left!')

#####CORRELATION MATRIX
trainMatrix = train.corr()
f, ax = plt.subplots(figsize=(57, 55))
sn.heatmap(trainMatrix, annot=True)
plt.savefig('graphs/heatmap_before.png')
plt.close

for col in combine_data.columns:
	if combine_data.dtypes[col] == 'object':
		combine_data = pd.concat([combine_data,pd.get_dummies(combine_data[col], prefix = col)],axis = 1)
		combine_data.drop(col, axis=1, inplace=True)

train = combine_data.loc[combine_data['Test'] == 0].copy()
train.drop('Test', axis=1, inplace=True)
test = combine_data.loc[combine_data['Test'] == 1].copy()
test.drop('Test', axis=1, inplace=True)
test.drop('SalePrice', axis=1, inplace=True)

#####REDEFINE DATA, TRAIN AND TEST NAME DUE TO CONCAT
train.name = 'train'
test.name = 'test'
data = [train,test]

binary_col_list = return_binary_col(train)

#####HISTOGRAMS CONTINUOUS DATA
f, ax = plt.subplots(4, 5, figsize=(20, 20))
l=1
for i in [element for element in continuous_list if element in train]:
	ax = plt.subplot(4,5,l)
	sn.histplot(data=train[i])
	ax.set_title(i)
	l+=1
f.tight_layout()
plt.savefig('graphs/hist_num.png')
plt.close

#####TRANSFORM CONTINUOUS DATA TO REDUCE SKEWNESS
transform_list=['SalePrice','LotArea','LotFrontage','GrLivArea','TotalBsmtSF','1stFlrSF','GarageArea','PorchSF']
for i in data:
	for col in [element for element in transform_list if element in train]:
		if i.name == 'test' and col == 'SalePrice':
			pass
		elif i.name == 'train' and col == 'SalePrice':
			i[col] = np.log1p(i.loc[:,col].values.reshape(-1, 1))
		else:
			i[col] = np.log1p(i.loc[:,col].values.reshape(-1, 1))

f, ax = plt.subplots(4, 5, figsize=(20, 20))
l=1
for i in [element for element in continuous_list if element in train]:
	ax = plt.subplot(4,5,l)
	sn.histplot(data=train[i])
	ax.set_title(i)
	l+=1
f.tight_layout()
plt.savefig('graphs/hist_num_quantile.png')
plt.close

f, ax = plt.subplots(4, 5, figsize=(20, 20))
l=1
for i in [element for element in continuous_list if element in train]:
	ax = plt.subplot(4,5,l)
	sn.boxplot(data=train[i])
	ax.set_title(i)
	l+=1
plt.savefig('graphs/box_num_quantile.png')
plt.close

#####DELETE DEFINE COLUMNS
for i in data:
	for col in bath_list:
		i.drop(col, axis=1, inplace=True)
	for col in garage_list:
		try:
			i.drop(col, axis=1, inplace=True)
		except:
			pass
	i = i.rename({'OpenPorchSF': 'OpenPorch', 'WoodDeckSF': 'WoodDeck'}, axis=1)

#####ANALYZE CORRELATION BETWEEN FEATURES; REMOVE HIGH CORRELATION FEATURES
high_corr, drop_corr = correlation_columns(train, 'SalePrice', 0.7, 0.95)

for col in drop_corr:
	train.drop(col, axis=1, inplace=True)
	test.drop(col, axis=1, inplace=True)

print('\nThe follwoing features show a high correlation! Check whether one of them can  be removed.')
print(*high_corr, sep = '\n')

print('\nAutomatically removed:')
print(*drop_corr, sep = '\n')

#####SKEWNESS OF TRANSFORMED FEATURES
cont_skew = {}
for col in [element for element in transform_list if element in train]:
	cont_skew[col] = train[col].skew()
print('\nSkewness of continuous features:')
[print(key,':',value) for key, value in cont_skew.items()]

#####CHECK FOR OUTLIERS
cont_outlier = count_outliers(train, transform_list, 3)
print('\nOutliers in continuous features (3*IQR):')
[print(key,':',value) for key, value in cont_outlier.items()]

#####SCATTER PLOT CONTINUOUS FEATURES
f, ax = plt.subplots(4, 4, figsize=(20, 20))
l=1
for col in [element for element in continuous_list if element in train]:
	ax = plt.subplot(4,4,l)
	sn.scatterplot(x=col,y='SalePrice',data=train)
	ax.set_title(col)
	l+=1
f.tight_layout()
plt.savefig('graphs/scatter_price_cont_final.png')
plt.close

#####REMOVE SOME MAX OUTLIERS - JUDGEMENT BASED ON SCATTER PLOT
outlier_col = ['LotArea', 'LotFrontage', 'MasVnrArea', 'GrLivArea', 'TotalBsmtSF']
outlier_row = []
for col in outlier_col:
	if train[col].idxmax() in outlier_row:
		pass
	else:
		outlier_row.append(train[col].idxmax())
	if col in ('GrLivArea'):
		print('\nMaximal values of %s:' % col)
		print(train[col].nlargest(2))
		second_max = train.loc[train[col] == train[col].nlargest(2).iloc[1]].index[0]
		if second_max in outlier_row:
			pass
		else:
			outlier_row.append(second_max)
print('\nRows with max value outliers:')
print(outlier_row)
train.drop(train.index[outlier_row], axis=0, inplace=True)

#####FINAL HISTOGRAMS SHWOING DISTRIBUTION OF CONTINUOUS FEATURES
f, ax = plt.subplots(4, 4, figsize=(20, 20))
l=1
for col in [element for element in continuous_list if element in train]:
	ax = plt.subplot(4,4,l)
	sn.histplot(data=train[col],kde=True)
	ax.set_title(col)
	l+=1
f.tight_layout()
plt.savefig('graphs/hist_after_distr.png')
plt.close

#####FINAL CORRELATION MATRIX
trainMatrix = train[list(set(train.columns) - set(binary_col_list))].corr()
f, ax = plt.subplots(figsize=(32, 30))
sn.heatmap(trainMatrix, annot=True)
plt.savefig('graphs/heatmap_cutoff.png')
plt.close

#####IDENTIFY FEATURES WITH LOW CORRELATION TO PRICE FOR REMOVAL BEFORE LINEAR REGRESSION
print('\n***Remove features with low correlation to SalePrice for linear regression***')
low_corr_dict = low_corr_target(train, 'SalePrice', 0.1)
low_corr = list(low_corr_dict.keys())
if 'Id' in low_corr:
	low_corr.remove('Id')
print(low_corr)

#####TRAIN AND TEST DATA, SPLIT
X_train = train.drop(['SalePrice', 'Id'], axis=1)
Y_train = train['SalePrice']
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=666)

print('\n***Performance of various models***')
#####LINEAR REGRESSION
x_train_lin = x_train.drop(low_corr, axis=1)
x_test_lin = x_test.drop(low_corr, axis=1)
linreg = LinearRegression()
linreg.fit(x_train_lin, y_train)
Y_pred = linreg.predict(x_test_lin)
print('Accuracy Linear Regression (RMSLE):',np.sqrt(metrics.mean_squared_log_error(np.expm1(y_test), np.expm1(Y_pred))))
#####RANDOM FOREST REGRESSOR
rf_reg=RandomForestRegressor(n_estimators= 1000, random_state=666)
rf_reg.fit(x_train,y_train)
Y_pred=rf_reg.predict(x_test)
print('Accuracy Random Forest (RMSLE):',np.sqrt(metrics.mean_squared_log_error(np.expm1(y_test), np.expm1(Y_pred))))

#####OPTIMIZED RANDOM FOREST REGRESSOR
# estimator = RandomForestRegressor(
# 	n_jobs=-1,
# 	random_state=666,
# )

# search_space = {
# 	'min_samples_split': (2, 15),
# 	'min_samples_leaf': (2, 15),
# 	'max_depth': (2, 15),
# 	'max_features': ['auto', 'sqrt', 'log2', None],
# 	'n_estimators': (5, 5000),
# }

# cv = KFold(n_splits=5, shuffle=True)
# n_iterations = 100
# bayes_cv_tuner = BayesSearchCV(
# 	estimator=estimator,
# 	search_spaces=search_space,
# 	scoring='neg_root_mean_squared_error',
# 	cv=cv,
# 	n_jobs=-1,
# 	n_iter=n_iterations,
# 	verbose=0,
# 	refit=True,
# )

# rf_reg_opt = bayes_cv_tuner.fit(x_train, y_train, callback=print_status)
# print('\nOptimized Random Forest Parameters:')
# print(rf_reg_opt.best_params_)
# Y_pred = rf_reg_opt.predict(x_test)
# print('Accuracy Optimized Random Forest (RMSLE):',np.sqrt(metrics.mean_squared_log_error(np.expm1(y_test), np.expm1(Y_pred))))
## Optimized Random Forest Classifier: Optimized Random Forest Parameters: ('max_depth', 13), ('max_features', None), ('min_samples_leaf', 2), ('min_samples_split', 2), ('n_estimators', 3165)

#####XGBoost REGRESSOR
xg_reg = xgb.XGBRegressor(colsample_bytree = 0.3, learning_rate = 0.05,max_depth = 5, n_estimators = 1000)
xg_reg.fit(x_train,y_train)
Y_pred = xg_reg.predict(x_test)
print('Accuracy XGBoost (RMSLE):',np.sqrt(metrics.mean_squared_log_error(np.expm1(y_test), np.expm1(Y_pred))))
f,ax = plt.subplots(figsize=(10,10))
xgb.plot_importance(xg_reg,ax=ax,color='red')
plt.savefig('graphs/feature_contribution_xgboost.png')
plt.close

#####OPTIMIZED XGBoost REGRESSOR
# estimator = xgb.XGBRegressor(
#     n_jobs=-1,
#     verbosity=0,
# )

# search_space = {
#     'learning_rate': (Real(0.01, 1.0, 'log-uniform')),
#     'eta': (Real(0.01, 0.3, 'log-uniform')),
#     'gamma': (0.0, 0.5),
#     'min_child_weight': (0, 10),
#     'max_depth': (2, 10),
#     'colsample_bytree': (Real(0.01, 1.0, 'log-uniform')),
#     'min_child_weight': (0, 5),
#     'reg_lambda': (Real(0.00001,10,'log-uniform')),
#     'reg_alpha': (Real(0.00001,10,'log-uniform')),
#     'subsample': (0.5, 1.0),
#     'n_estimators': (5, 5000),
# }

# cv = KFold(n_splits=5, shuffle=True)
# n_iterations = 100
# bayes_cv_tuner = BayesSearchCV(
#     estimator=estimator,
#     search_spaces=search_space,
#     scoring='neg_root_mean_squared_error',
#     cv=cv,
#     n_jobs=-1,
#     n_iter=n_iterations,
#     verbose=0,
#     refit=True,
# )

# xg_reg_opt = bayes_cv_tuner.fit(x_train, y_train, callback=print_status)
# print('\nOptimized XGBoost Parameters:')
# print(xg_reg_opt.best_params_)
# Y_pred = xg_reg_opt.predict(x_test)
# print('Accuracy Optimized XGBoost (RMSLE):',np.sqrt(metrics.mean_squared_log_error(np.expm1(y_test), np.expm1(Y_pred))))
## Best parameters XGBoost: ('colsample_bytree', 0.08652163908865458), ('eta', 0.29999999999999993), ('gamma', 0.0), ('learning_rate', 0.0296450572630713), ('max_depth', 3), ('min_child_weight', 4), ('n_estimators', 995)

#####CatBoost REGRESSOR
cb_reg = cbr.CatBoostRegressor(n_estimators = 1000)
cb_reg.fit(x_train,y_train,silent=True)
Y_pred = cb_reg.predict(x_test)
print('Accuracy CatBoost (RMSLE):',np.sqrt(metrics.mean_squared_log_error(np.expm1(y_test), np.expm1(Y_pred))))

####OPTIMIZED CatBoost REGRESSOR
# estimator = cbr.CatBoostRegressor(
#     silent=True,
# )

# search_space = {
#     'learning_rate': (Real(0.01, 0.3, 'log-uniform')),
#     'max_depth': (3, 10),
#     'l2_leaf_reg': (0.2, 30),
#     'iterations': (20, 2000),
# }

# cv = KFold(n_splits=3, shuffle=True)
# n_iterations = 50
# bayes_cv_tuner = BayesSearchCV(
#     estimator=estimator,
#     search_spaces=search_space,
#     scoring='neg_root_mean_squared_error',
#     cv=cv,
#     n_jobs=-1,
#     n_iter=n_iterations,
#     verbose=0,
#     refit=True,
# )

# cb_reg_opt = bayes_cv_tuner.fit(x_train, y_train, callback=print_status)
# print('\nOptimized CatBoost Parameters:')
# print(cb_reg_opt.best_params_)
# Y_pred = cb_reg_opt.predict(x_test)
# print('Accuracy Optimized CatBoost (RMSLE):',np.sqrt(metrics.mean_squared_log_error(np.expm1(y_test), np.expm1(Y_pred))))
## Best parameters: ('iterations', 1875), ('l2_leaf_reg', 30), ('learning_rate', 0.022068113971935165), ('max_depth', 3)

#####PREDICT TEST DATA AND EXPORT FOR UPLOAD
predict_export = pd.DataFrame()
predict_export['Id'] = test['Id']

prediction_catboost = cb_reg.predict(test.drop('Id', axis=1))
prediction_catboost = np.expm1(prediction_catboost)
predict_export['SalePrice'] = prediction_catboost
predict_export.to_csv('submission.csv',index=False)