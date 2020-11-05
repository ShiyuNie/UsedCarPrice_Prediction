
import numpy as np 
import math
import scipy as sc
from copy import deepcopy
from dateutil import rrule
import datetime
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
#matplotlib inline
#plt.rcParams['figure.figsize'] = (20, 1000)
#plt.style.use('ggplot')
import pprint
plt.rcParams['axes.unicode_minus']=False

from sklearn import preprocessing
from sklearn.preprocessing import QuantileTransformer,StandardScaler,RobustScaler
from sklearn.model_selection import train_test_split,KFold,GridSearchCV,RandomizedSearchCV,cross_val_score
from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor,ExtraTreesRegressor,RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor 
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge,Lasso,ElasticNet
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from mlxtend.regressor import StackingCVRegressor
from scipy.special import inv_boxcox1p


# 查看数据主要信息
def lookinto_data(data):
	data.info() # 数据信息
	print(data.shape) # 查看数据的行列大小
	print(data.head(30))  # 查看前30行
	print(data.describe([0.25,0.50,0.75,0.99])) # 查看数据统计量







if __name__ == "__main__":

	################################## 读取清洗后的数据并归一化
	filein0 = 'train_clean_new.csv'
	filein1 = 'test_clean_new.csv'
	train = pd.read_csv(filein0, sep=',', encoding='utf-8')
	test = pd.read_csv(filein1, sep=',', encoding='utf-8')
	#lookinto_data(train)
	#lookinto_data(test)
	cols = train.columns
	feat_cols = [x for x in cols if x != 'price']
	train_target = train['price']
	train_feat = train[feat_cols]
	#print(feat_cols)
	'''
	# 随机采样70%数据构建训练样本，30%为测试样本
	X_train, X_test, y_train, y_test = train_test_split(train_feat, train_target, random_state=0, test_size=0.3) # 分层抽样 stratify=y_train
	# 归一化
	X_scaler = RobustScaler().fit(X_train)
	y_scaler = RobustScaler().fit(y_train.values.reshape(-1,1))
	X_train = X_scaler.transform(X_train)
	X_test = X_scaler.transform(X_test)
	y_train = y_scaler.transform(y_train.values.reshape(-1,1))
	y_test = y_scaler.transform(y_test.values.reshape(-1,1))'''
	scores = {}
	train_predict = {}
	test_predict = {}
	# 交叉验证k折
	kfold= KFold(n_splits=10, random_state=10, shuffle=True)


	
	################################## 岭回归，网格搜索优化
	'''
	Ridge0 = Ridge()
	param_grid = {"alpha":[10.0, 11.0, 12.0, 13.0, 13.5, 14.0, 14.5, 15.0, 16.0]}
	grid_Ridge = GridSearchCV(Ridge0, cv=kfold, param_grid=param_grid,scoring='neg_mean_absolute_error')
	grid_Ridge.fit(X_train, y_train)
	best_para = grid_Ridge.best_params_
	Ridge_score = grid_Ridge.best_score_
	#print(best_para['alpha'],Ridge_score)		# 14.0 -0.115914144345942
	# 最优回归
	#Ridge_best = Ridge(alpha=best_para['alpha'])
	Ridge_best = Ridge(alpha=14.0)
	Ridge_best.fit(X_train,y_train)
	Ridge_train_predict = Ridge_best.predict(X_train)
	Ridge_test_predict = Ridge_best.predict(X_test)
	Ridge_train_predict = Ridge_train_predict.tolist()
	Ridge_test_predict = Ridge_test_predict.tolist()
	Ridge_train_predict = [i for item in Ridge_train_predict for i in item]
	Ridge_test_predict = [i for item in Ridge_test_predict for i in item]
	# MAE
	MAE_train = mean_absolute_error(y_train, Ridge_train_predict)
	MAE_test = mean_absolute_error(y_test, Ridge_test_predict)
	#print(MAE_train,MAE_test)
	# 0.1158559633429613 0.11659014660877351
	scores['Ridge'] = [MAE_train,MAE_test]
	train_predict['Ridge'] = Ridge_train_predict
	test_predict['Ridge'] = Ridge_test_predict
	#print(Ridge_train_predict,Ridge_test_predict)'''

	
	################################## Lasso回归，网格搜索优化
	'''
	Lasso0 = Lasso()
	param_grid = {"alpha":[0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.001, 0.1]}
	grid_Lasso = GridSearchCV(Lasso0, cv=kfold, param_grid=param_grid,scoring='neg_mean_absolute_error')
	grid_Lasso.fit(X_train, y_train)
	best_para = grid_Lasso.best_params_
	Lasso_score = grid_Lasso.best_score_
	#print(best_para['alpha'],Lasso_score)		# 0.0003 -0.1156808674899257
	# 最优回归
	#Lasso_best = Lasso(alpha=best_para['alpha'])
	Lasso_best = Lasso(alpha=0.0003)
	Lasso_best.fit(X_train,y_train)
	Lasso_train_predict = Lasso_best.predict(X_train)
	Lasso_test_predict = Lasso_best.predict(X_test)
	# MAE
	MAE_train = mean_absolute_error(y_train, Lasso_train_predict)
	MAE_test = mean_absolute_error(y_test, Lasso_test_predict)
	#print(MAE_train,MAE_test)
	# 0.11563943977937828 0.1163896756213416
	scores['Lasso'] = [MAE_train,MAE_test]
	train_predict['Lasso'] = list(Lasso_train_predict)
	test_predict['Lasso'] = list(Lasso_test_predict)
	#print(train_predict['Lasso'],test_predict['Lasso'])'''

	
	################################## ElasticNet回归，网格搜索优化
	'''
	EN0 = ElasticNet()
	param_grid = {"alpha":[0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]}
	grid_EN = GridSearchCV(EN0, cv=kfold, param_grid=param_grid,scoring='neg_mean_absolute_error')
	grid_EN.fit(X_train, y_train)
	best_para = grid_EN.best_params_
	EN_score = grid_EN.best_score_
	print(best_para['alpha'],EN_score)		# 0.0002 -0.1157431885951446
	# 最优回归
	#EN_best = ElasticNet(alpha=best_para['alpha'])
	EN_best = ElasticNet(alpha=0.0002)
	EN_best.fit(X_train,y_train)
	EN_train_predict = EN_best.predict(X_train)
	EN_test_predict = EN_best.predict(X_test)
	# MAE
	MAE_train = mean_absolute_error(y_train, EN_train_predict)
	MAE_test = mean_absolute_error(y_test, EN_test_predict)
	#print(MAE_train,MAE_test)
	# 0.1156919843166006 0.11644348964778631
	scores['ElasticNet'] = [MAE_train,MAE_test]
	train_predict['ElasticNet'] = list(EN_train_predict)
	test_predict['ElasticNet'] = list(EN_test_predict)
	print(EN_train_predict.shape,EN_test_predict.shape)'''


	################################## SupportVector回归，随机搜索优化
	'''
	SVR0 = SVR()
	param_rand = {"kernel":['rbf','sigmoid','poly'],
				"gamma":[0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001],
				"coef0":[-0.0001,-0.0005,-0.001,-0.005,-0.01,-0.05],
				"C":sc.stats.randint(10,50),
				"epsilon":[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01],
				"max_iter":[90,100]}
	rand_SVR = RandomizedSearchCV(SVR0, cv=kfold, param_distributions=param_rand, scoring='neg_mean_absolute_error',n_iter=20)
	rand_SVR.fit(X_train, y_train.ravel())
	best_para = rand_SVR.best_params_
	SVR_score = rand_SVR.best_score_
	print(best_para,SVR_score)		# epsilon': 0.001, 'kernel': 'sigmoid', 'max_iter': 100, 'C': 20, 'coef0': -0.01, 'gamma': 0.0006'''
	# 最优回归
	'''
	SVR_best = SVR(kernel='sigmoid', gamma=0.0006, coef0=-0.01, C=20, epsilon=0.001, max_iter=100)
	SVR_best.fit(X_train,y_train.ravel())
	SVR_train_predict = SVR_best.predict(X_train)
	SVR_test_predict = SVR_best.predict(X_test)
	# MAE
	MAE_train = mean_absolute_error(y_train, SVR_train_predict)
	MAE_test = mean_absolute_error(y_test, SVR_test_predict)
	#print(MAE_train,MAE_test)
	# 0.20644136234032773 0.20623639098523264
	scores['SupportVector'] = [MAE_train,MAE_test]
	train_predict['SupportVector'] = list(SVR_train_predict)
	test_predict['SupportVector'] = list(SVR_test_predict)
	print(SVR_train_predict.shape,SVR_test_predict.shape)'''


	################################## GradientBoosting 回归，网格搜索优化
	'''
	GBR0 = GradientBoostingRegressor(loss='huber', alpha=0.8, learning_rate=0.1, n_estimators=100, subsample=0.9,
		min_samples_split=100, min_samples_leaf=100, max_features='sqrt', warm_start=True)
	param_grid = {"max_depth":[5,10,20]}
	grid_GBR = GridSearchCV(GBR0, cv=kfold, param_grid=param_grid, scoring='neg_mean_absolute_error')
	grid_GBR.fit(X_train, y_train.ravel())
	best_para = grid_GBR.best_params_
	GBR_score = grid_GBR.best_score_
	print(best_para,GBR_score)	# maxdepth10 -0.09145699440230878'''
	
	# 最优回归
	GBR_best = GradientBoostingRegressor(loss='huber', alpha=0.8, learning_rate=0.1, n_estimators=300, subsample=0.9,
		min_samples_split=100, min_samples_leaf=100, max_depth=10, max_features='sqrt', warm_start=True)
	'''
	GBR_best.fit(X_train,y_train.ravel())
	GBR_train_predict = GBR_best.predict(X_train)
	GBR_test_predict = GBR_best.predict(X_test)
	# MAE
	MAE_train = mean_absolute_error(y_train, GBR_train_predict)
	MAE_test = mean_absolute_error(y_test, GBR_test_predict)
	#print(MAE_train,MAE_test)
	# 0.06712594366922428 0.08063293090175423
	scores['GBDT'] = [MAE_train,MAE_test]
	train_predict['GBDT'] = list(GBR_train_predict)
	test_predict['GBDT'] = list(GBR_test_predict)
	print(GBR_train_predict.shape,GBR_test_predict.shape)'''

	'''
	################################## eXtremeGradientBoosting 回归，网格搜索优化
	XGBR0 = XGBRegressor(learning_rate=0.1, n_estimators=300, gamma=0, max_depth=10, min_child_weight=5, subsample=0.9, 
	colsample_bytree=0.9, reg_alpha=0.001, reg_lambda=0.1, objective='reg:squarederror')
	param_grid = {"reg_lambda":[0.001,0.01,0.05,0.1,0.5,1]}
	grid_XGBR = GridSearchCV(XGBR0, cv=kfold, param_grid=param_grid, scoring='neg_mean_absolute_error')
	grid_XGBR.fit(X_train, y_train.ravel())
	best_para = grid_XGBR.best_params_
	XGBR_score = grid_XGBR.best_score_
	print(best_para,XGBR_score)	# -0.07334190887225099
	
	# 最优回归
	XGBR_best = XGBRegressor(learning_rate=0.1, n_estimators=300, gamma=0, max_depth=10, min_child_weight=5, subsample=0.9, 
	colsample_bytree=0.9, reg_alpha=0.001, reg_lambda=0.1, objective='reg:squarederror')
	XGBR_best.fit(X_train,y_train.ravel())
	XGBR_train_predict = XGBR_best.predict(X_train)
	XGBR_test_predict = XGBR_best.predict(X_test)
	# MAE
	MAE_train = mean_absolute_error(y_train, XGBR_train_predict)
	MAE_test = mean_absolute_error(y_test, XGBR_test_predict)
	#print(MAE_train,MAE_test)
	# 0.036141258103028345 0.07161456456799196
	scores['XGB'] = [MAE_train,MAE_test]
	train_predict['XGB'] = list(XGBR_train_predict)
	test_predict['XGB'] = list(XGBR_test_predict)
	print(XGBR_train_predict.shape,XGBR_test_predict.shape)'''

	
	'''
	################################## ExtraTree 回归，网格搜索优化
	ETR0 = ExtraTreesRegressor(n_estimators=70, max_features='auto', max_depth=10, bootstrap=False,
		min_samples_split=150, min_samples_leaf=5) # , criterion='mae'
	param_grid = {"bootstrap":[False,True],
					"min_samples_leaf":[5,10,30,50]}
	grid_ETR = GridSearchCV(ETR0, cv=kfold, param_grid=param_grid, scoring='neg_mean_absolute_error')
	grid_ETR.fit(X_train, y_train.ravel())
	best_para = grid_ETR.best_params_
	ETR_score = grid_ETR.best_score_
	print(best_para,ETR_score)	# -0.11489663875633793
	
	# 最优回归
	ETR_best = ExtraTreesRegressor(n_estimators=70, max_features='auto', max_depth=10, bootstrap=False,
		min_samples_split=150, min_samples_leaf=5)
	ETR_best.fit(X_train,y_train.ravel())
	ETR_train_predict = ETR_best.predict(X_train)
	ETR_test_predict = ETR_best.predict(X_test)
	# MAE
	MAE_train = mean_absolute_error(y_train, ETR_train_predict)
	MAE_test = mean_absolute_error(y_test, ETR_test_predict)
	print(MAE_train,MAE_test)
	# 0.11495342318487849 0.11653832428037571
	scores['ExtraTree'] = [MAE_train,MAE_test]
	train_predict['ExtraTree'] = list(ETR_train_predict)
	test_predict['ExtraTree'] = list(ETR_test_predict)
	#print(ETR_train_predict.shape,ETR_test_predict.shape)'''

	'''
	################################## RandomForest 回归，网格搜索优化
	RFR0 = RandomForestRegressor(n_estimators=50, max_features='auto', max_depth=10, bootstrap=True,
		min_samples_split=50, min_samples_leaf=10) 
	param_grid = {"max_depth":[3,5,8,10]}
	grid_RFR = GridSearchCV(RFR0, cv=kfold, param_grid=param_grid, scoring='neg_mean_absolute_error')
	grid_RFR.fit(X_train, y_train.ravel())
	best_para = grid_RFR.best_params_
	RFR_score = grid_RFR.best_score_
	print(best_para,RFR_score)	# -0.10186739132131446
	
	# 最优回归
	RFR_best = RandomForestRegressor(n_estimators=50, max_features='auto', max_depth=10, bootstrap=True,
		min_samples_split=50, min_samples_leaf=10) 
	RFR_best.fit(X_train,y_train.ravel())
	RFR_train_predict = RFR_best.predict(X_train)
	RFR_test_predict = RFR_best.predict(X_test)
	# MAE
	MAE_train = mean_absolute_error(y_train, RFR_train_predict)
	MAE_test = mean_absolute_error(y_test, RFR_test_predict)
	print(MAE_train,MAE_test)
	# 0.09730358466242835 0.10178361141941208
	scores['RandomForest'] = [MAE_train,MAE_test]
	train_predict['RandomForest'] = list(RFR_train_predict)
	test_predict['RandomForest'] = list(RFR_test_predict)
	#print(ETR_train_predict.shape,ETR_test_predict.shape)'''

	'''
	################################## LightGradientBoosting 回归，网格搜索优化
	LGBR0 = LGBMRegressor(objective='regression', metric='rmse', learning_rate=0.1, n_estimators=300, 
		max_depth=10, num_leaves=150, max_bin=200, min_child_samples=200, min_child_weight=0.001, 
	bagging_freq=3, bagging_fraction=0.9, feature_fraction=0.9, 
	lambda_l1=0.5, lambda_l2=0.01)
	param_grid = {"learning_rate":[0.1, 0.01, 0.05, 0.2, 0.07],
	"n_estimators":[100,200,50,80,300]}
	grid_LGBR = GridSearchCV(LGBR0, cv=kfold, param_grid=param_grid, scoring='neg_mean_absolute_error')
	grid_LGBR.fit(X_train, y_train.ravel())
	best_para = grid_LGBR.best_params_
	LGBR_score = grid_LGBR.best_score_
	print(best_para,LGBR_score)	# -0.07313867983412776
	
	# 最优回归
	LGBR_best = LGBMRegressor(objective='regression', metric='rmse', learning_rate=0.01, n_estimators=6000, 
		max_depth=10, num_leaves=150, max_bin=200, min_child_samples=200, min_child_weight=0.001, 
		bagging_freq=3, bagging_fraction=0.9, feature_fraction=0.9, lambda_l1=0.5, lambda_l2=0.01)
	LGBR_best.fit(X_train,y_train.ravel())
	LGBR_train_predict = LGBR_best.predict(X_train)
	LGBR_test_predict = LGBR_best.predict(X_test)
	# MAE
	MAE_train = mean_absolute_error(y_train, LGBR_train_predict)
	MAE_test = mean_absolute_error(y_test, LGBR_test_predict)
	print(MAE_train,MAE_test)
	# 0.05482467175655363 0.06870844382749636
	scores['LGBM'] = [MAE_train,MAE_test]
	train_predict['LGBM'] = list(LGBR_train_predict)
	test_predict['LGBM'] = list(LGBR_test_predict)'''
	#print(LGBR_train_predict.shape,LGBR_test_predict.shape)


	################################## Stacking回归
	'''
	SR = StackingCVRegressor(regressors=(GBR_best, XGBR_best, LGBR_best), 
		meta_regressor=XGBR_best, use_features_in_secondary=True)
	# Ridge_best, Lasso_best, EN_best, ETR_best, RFR_best,
	SR.fit(X_train,y_train.ravel())
	SR_train_predict = SR.predict(X_train)
	SR_test_predict = SR.predict(X_test)
	# MAE
	MAE_train = mean_absolute_error(y_train, SR_train_predict)
	MAE_test = mean_absolute_error(y_test, SR_test_predict)
	print('stacking',MAE_train,MAE_test)
	# 0.03713471030012472 0.07001761883255012
	scores['Stacking'] = [MAE_train,MAE_test]
	train_predict['Stacking'] = list(SR_train_predict)
	test_predict['Stacking'] = list(SR_test_predict)'''
	#print(SR_train_predict.shape,SR_test_predict.shape)
	#print(train_predict,test_predict)
	
	################################## 混合Blending模型
	'''
	scores = pd.DataFrame(scores)
	train_predict = pd.DataFrame(train_predict)
	test_predict = pd.DataFrame(test_predict)
	scores.sort_index(axis=1,inplace=True)
	train_predict.sort_index(axis=1,inplace=True)
	test_predict.sort_index(axis=1,inplace=True)
	print(scores)
	print(train_predict.shape,test_predict.shape)'''
	'''
	# Blending混合模型
	n_train = train_predict.shape[0]
	n_test = test_predict.shape[0]
	n = train_predict.shape[1]
	print(n_train,n_test,n)

	Blend_train_predict = [0 for i in range(n_train)]
	Blend_test_predict = [0 for i in range(n_test)]
	for j in range(n_train):
		Blend_train_predict[j] = np.mean(train_predict.iloc[j,:])
	for k in range(n_test):
		Blend_test_predict[k] = np.mean(test_predict.iloc[k,:])
	#print(Blend_train_predict,Blend_test_predict)
	# MAE
	MAE_train = mean_absolute_error(y_train, Blend_train_predict)
	MAE_test = mean_absolute_error(y_test, Blend_test_predict)
	print('blend',MAE_train,MAE_test)
	scores['Blend'] = [MAE_train,MAE_test]
	print(scores)	# 0.043684  0.067320'''


	'''
	####################################### 画图score
	fig = plt.figure(figsize=(10, 8))
	y1 = scores.iloc[0,:]
	y2 = scores.iloc[1,:]
	x = [i for i in range(len(y1))]
	plt.plot(x, y1, marker='o', linestyle='-', color='blue', label='train')
	plt.plot(x, y2, marker='o', linestyle='-', color='cyan', label='test')
	for a,b in zip(x,y1):
		plt.text(a, b+0.00005, '%.6f' % b, horizontalalignment='left', size='small', color='black')
	for a,b in zip(x,y2):
		plt.text(a, b-0.00005, '%.6f' % b, horizontalalignment='right', size='small', color='black')
	x_t = list(scores.columns)
	plt.xticks(x, x_t, color='black')
	plt.ylabel('Score (MAE)', size=20, labelpad=10)
	plt.xlabel('Model', size=20, labelpad=10)
	plt.tick_params(axis='x', labelsize=10)
	plt.tick_params(axis='y', labelsize=10)
	plt.legend(loc='upper left')
	plt.title('Scores of Models', size=15)
	plt.show()
	'''

	
	############################################### 根据以上选择最优回归模型
	# 归一化
	X_scaler = RobustScaler().fit(train_feat)
	y_scaler = RobustScaler().fit(train_target.values.reshape(-1,1))
	X_train = X_scaler.transform(train_feat)
	y_train = y_scaler.transform(train_target.values.reshape(-1,1))
	test_feat = X_scaler.transform(test)

	'''# No1 submit			score 566.48
	XGBR_best = XGBRegressor(learning_rate=0.1, n_estimators=300, gamma=0, max_depth=10, min_child_weight=5, subsample=0.9, 
	colsample_bytree=0.9, reg_alpha=0.001, reg_lambda=0.1, objective='reg:squarederror')
	XGBR_best.fit(X_train,y_train.ravel())
	XGBR_train_predict = XGBR_best.predict(X_train)
	XGBR_test_predict = XGBR_best.predict(test_feat)
	# MAE
	MAE_train = mean_absolute_error(y_train, XGBR_train_predict)
	print(MAE_train)
	predict describe
		mean    5823.089660
		std     7283.779292
		min       13.000000
		25%     1315.750000
		50%     3200.000000
		75%     7528.500000
		99%    34596.020000
		max    94269.000000'''

	'''# No2 submit			score 532.37
	SR.fit(X_train,y_train.ravel())
	SR_train_predict = SR.predict(X_train)
	SR_test_predict = SR.predict(test_feat)
	# MAE
	MAE_train = mean_absolute_error(y_train, SR_train_predict)
	print(MAE_train)
			           price
			count     50000.000000
			mean      5843.024019
			std       7371.555618
			min         14.742720
			25%       1317.373260
			50%       3178.293945
			75%       7537.399414
			99%      35291.151992
			max      95132.281250'''


	# No3 submit Blending混合模型	score 530.40
	XGBR_best = XGBRegressor(learning_rate=0.1, n_estimators=300, gamma=0, max_depth=10, min_child_weight=5, subsample=0.9, 
	colsample_bytree=0.9, reg_alpha=0.001, reg_lambda=0.1, objective='reg:squarederror')
	XGBR_best.fit(X_train,y_train.ravel())
	XGBR_test_predict = XGBR_best.predict(test_feat)
	test_predict['XGB'] = list(XGBR_test_predict)

	LGBR_best = LGBMRegressor(objective='regression', metric='rmse', learning_rate=0.01, n_estimators=6000, 
		max_depth=10, num_leaves=150, max_bin=200, min_child_samples=200, min_child_weight=0.001, 
		bagging_freq=3, bagging_fraction=0.9, feature_fraction=0.9, lambda_l1=0.5, lambda_l2=0.01)
	LGBR_best.fit(X_train,y_train.ravel())
	LGBR_test_predict = LGBR_best.predict(test_feat)
	test_predict['LGBM'] = list(LGBR_test_predict)

	SR = StackingCVRegressor(regressors=(GBR_best, XGBR_best, LGBR_best), 
		meta_regressor=XGBR_best, use_features_in_secondary=True)
	# Ridge_best, Lasso_best, EN_best, ETR_best, RFR_best,
	SR.fit(X_train,y_train.ravel())
	SR_test_predict = SR.predict(test_feat)
	test_predict['Stacking'] = list(SR_test_predict)

	test_predict = pd.DataFrame(test_predict)
	test_predict.sort_index(axis=1,inplace=True)
	print(test_predict.shape)

	n_test = test_predict.shape[0]
	n = test_predict.shape[1]
	print(n_test,n)

	Blend_test_predict = [0 for i in range(n_test)]
	for k in range(n_test):
		Blend_test_predict[k] = np.mean(test_predict.iloc[k,:])
	#print(Blend_train_predict,Blend_test_predict)

	# 输出预测
	Blend_test_predict = np.array(Blend_test_predict)
	test_predict = y_scaler.inverse_transform(Blend_test_predict.reshape(-1, 1))
	lam = 0.08080808080808081
	test_predict = list(inv_boxcox1p(test_predict, lam))
	test_predict = [i for item in test_predict for i in item]
	#print(test_predict)

	predict = pd.DataFrame(data={'SaleID':np.arange(150000, 150000+len(test_predict)),'price':test_predict})
	predict.to_csv('submission3.csv', index=False, header=True, encoding='utf_8')
	
	for i in range(len(test_predict)):
		if test_predict[i]<=0:
			print('false',test_predict[i])
	#print(predict[:5].to_csv())
	print(predict.describe([0.25,0.50,0.75,0.99]))
	'''count     50000.000000
		mean    5828.622363
		std       7313.397135
		min       14.491944
		25%       1311.051883
		50%       3187.877393
		75%      7553.023001
		99%      34972.869525
		max      90843.849867
			train describe
		count  150000.000000
		mean     5923.327333
		std      7501.998477
		min        11.000000
		25%      1300.000000
		50%      3250.000000
		75%      7700.000000
		99%     34950.000000
		max     99999.000000'''

