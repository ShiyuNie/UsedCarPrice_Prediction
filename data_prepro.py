
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
from sklearn.preprocessing import QuantileTransformer,StandardScaler,MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error


# 查看数据主要信息
def lookinto_data(data):
	data.info() # 数据信息
	print(data.shape) # 查看数据的行列大小
	print(data.head(30))  # 查看前30行
	print(data.describe([0.25,0.50,0.75,0.99])) # 查看数据统计量


# 查看缺失值个数和所占百分比
def missing_data(data):
	missing_num = data[data.columns].isnull().sum(axis=0).sort_values(ascending=False)
	missing_per = (100*data[data.columns].isnull().sum()/len(data)).sort_values(ascending=False)
	missing_data = pd.concat([missing_num,missing_per],keys=['null_num','null_percentage'],axis=1)
	missing_data = missing_data[missing_data['null_percentage']>0]
	return missing_data


# 分别输出年月日
def read_ymd(date):
	year = [x//10000 for x in date]
	month = list(np.array([x//100 for x in date])-100*np.array(year))
	day = [x%100 for x in date]
	return year,month,day


# 画各列数据箱型图
def box_all(data):
	fig,axes = plt.subplots(3,8,figsize=(80,20))
	fig.subplots_adjust(hspace=0.6, wspace=0.3)
	for i,ax in zip(data.columns,axes.flatten()):
		ax.boxplot(x=i, data=data)
		ax.set_title(str(i),size=10)
		ax.grid(linestyle="--", alpha=0.3)
		ax.set_xticks([])
	plt.show()


# 画各列数据直方图
def hist_all(data):
	fig,axes = plt.subplots(2,14,figsize=(20,1000),sharey='row')
	fig.subplots_adjust(hspace=0.5, wspace=0.5)
	for i,ax in zip(data.columns,axes.flatten()):
		data[i].hist(bins=10, ax=ax)
		ax.set_title(str(i),size=8)
	plt.show()


# 输出各列异常值（z_score>3）
def get_outliers(data,cols):
	outliers = [[] for i in range(len(cols))]	# 异常值
	data_z = deepcopy(data)		# z得分
	for i in range(len(cols)):
		col = cols[i]
		data_col = data[col]		# 得到每列的值
		z_score = (data_col-data_col.mean())/data_col.std()		# 计算每列的z得分
		data_z[col] = z_score.abs()>3		# z得分大于3.0为异常值
		n = len(data[col])
	#print(data_z)
	for i in range(n):
		for j in range(len(cols)):
			if data_z.iloc[i,j] == True:
				outliers[j].append(data.iloc[i,j])		# 得到每列异常值
	return outliers


# 各连续变量间关系散点图
def scatter_v(data,cols):
	fig,axes = plt.subplots(2,2,figsize=(20,40))
	fig.subplots_adjust(hspace=0.6, wspace=0.6)
	for i,ax in zip(cols,axes.flatten()):
		a = [x for x in cols if x not in i]
		s = a[2]
		norm = plt.Normalize(data[s].min(), data[s].max())
		sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
		sm.set_array([])
		axn = sn.scatterplot(x=a[0], y=a[1], hue=a[2], data=data, ax=ax, palette="RdBu")
		plt.xlabel(a[0], fontsize=6)
		plt.ylabel(a[1], fontsize=6)
		fig.colorbar(sm,ax=ax)
	plt.show()


# 该年月份众数替换月份0
def regmonth_replace0(data):
	n = len(data['regDate'])
	regdate = data['regDate'].tolist()
	regyear,regmonth,regday = read_ymd(regdate)
	year = set(regyear)
	year = list(year)
	year.sort()
	n_year = len(year)
	#print(year,n_year)
	y_month = [[] for i in range(n_year)]
	for i in range(n):
		if regmonth[i] != 0:
			for j in range(n_year):
				if regyear[i] == year[j]:
					y_month[j].append(regmonth[i])		# 每年对应的所有月份
	modmonth = []
	for i in range(n_year):
		modmonth.append(sc.stats.mode(y_month[i])[0][0])	# 求每年月份众数
	#print(y_month,modmonth)
	for i in range(n):
		if regmonth[i] == 0:
			for j in range(n_year):
				if regyear[i] == year[j]:
					regmonth[i] = modmonth[j]		# 每年月份众数替换0月份
	return regmonth


# 两个日期间年差
def get_usedyear(regdate,creatdate):
	delta = []
	for i in range(len(creatdate)):
		d1_y = datetime.datetime.strptime(str(creatdate[i]), '%Y%m').year
		d1_m = datetime.datetime.strptime(str(creatdate[i]), '%Y%m').month
		d2_y = datetime.datetime.strptime(str(regdate[i]), '%Y%m').year
		d2_m = datetime.datetime.strptime(str(regdate[i]), '%Y%m').month
		delta.append((d1_y-d2_y) + (d1_m-d2_m)/12)
	#print(delta)
	return delta


# 各连续变量与价格散点图
def scatter_price(data,cols):
	fig,axes = plt.subplots(1,3,sharey=True,figsize=(10,20))
	fig.subplots_adjust(hspace=0.6, wspace=0.6)
	for i,ax in zip(cols,axes.flatten()):
		sn.scatterplot(x=i, y='price', hue='price', data=data, ax=ax, palette='viridis_r')
		plt.xlabel(i, fontsize=6)
		plt.ylabel('Price', fontsize=6)
		ax.set_title('Price'+' - '+str(i), size=10)
	plt.show()


# 各离散变量与价格箱型图
def box_price(data,cols):
	fig,axes = plt.subplots(2,2,sharey=True,figsize=(20,20))
	fig.subplots_adjust(hspace=0.6, wspace=0.3)
	for i,ax in zip(cols,axes.flatten()):
		sn.boxplot(x=i, y='price', data=data, width=0.5, ax=ax)
		#i = plt.xticks(rotation=90)
		plt.ylabel('Price', fontsize=6)
		plt.xlabel(i, fontsize=6)
		ax.set_title('Price'+' - '+str(i), size=10)
	plt.show()


# 各离散变量与价格violin图
def violin_price(data,cols):
	fig,axes = plt.subplots(2,2,sharey=True,figsize=(20,20))
	fig.subplots_adjust(hspace=0.6, wspace=0.3)
	for i,ax in zip(cols,axes.flatten()):
		sn.violinplot(x=i, y='price', data=data, inner="quartile", scale="count", width=0.5, ax=ax)
		#i = plt.xticks(rotation=90)
		plt.ylabel('Price', fontsize=6)
		plt.xlabel(i, fontsize=6)
		ax.set_title('Price'+' - '+str(i), size=10)
	plt.show()


# 计算data所有字段VIF
def get_vif(data):
	data_tem = deepcopy(data)
	data_tem['c'] = 1		# 添加常数项
	cols = data.columns
	x = np.matrix(data_tem)
	VIF_list = [variance_inflation_factor(x,i) for i in range(x.shape[1]-1)]
	VIF = pd.DataFrame({'feature':cols,"VIF":VIF_list})
	#VIF = VIF.drop('c', axis=0)
	return VIF


# 计算data所有字段相关系数（spearman），并画图
def get_correlation(data):
	correlation = np.round(data.corr(method='spearman'),2)
	cols = data.columns
	cor = pd.DataFrame(correlation, columns=cols)
	fig = plt.figure(figsize=(40,40), dpi= 60)
	sn.heatmap(correlation, xticklabels=data.corr().columns, yticklabels=data.corr().columns, cmap='RdYlGn', center=0, annot=True, square=True)
	plt.xticks(fontsize=6, rotation=90)
	plt.yticks(fontsize=6)
	plt.ylim(0,len(cor))
	fig.tight_layout()
	plt.show()
	return correlation





if __name__ == "__main__":

	################################## 读取数据，观察数据
	filein0 = 'used_car_train_20200313.csv'
	filein1 = 'used_car_testB_20200421.csv'
	data = pd.read_csv(filein0, sep=' ', encoding='utf-8')
	test = pd.read_csv(filein1, sep=' ', encoding='utf-8')
	#lookinto_data(data)
	#lookinto_data(test)

	# 查找重复值
	#data_c = deepcopy(data)
	#data_c.drop(['SaleID'],axis=1)
	#print(data_c.duplicated().any())		# FALSE

	# 查看含有缺失值的列
	nan_col0 = data.isnull().any()
	nan_col1 = test.isnull().any()
	#print(nan_col0,nan_col1)

	# 各列空值个数和百分比
	missing_train = missing_data(data)
	missing_test = missing_data(test)
	#print(missing_train)
	#print(missing_test)
	'''
		data
				null_num  null_percentage
	fuelType	  8680		 5.786667
	gearbox	   5981		 3.987333
	bodyType	  4506		 3.004000
	model			1		 0.000667
		test
				null_num  null_percentage
	fuelType	  2924			5.848
	gearbox	   1968			3.936
	bodyType	  1504			3.008
	'''

	################################### 查看各列数据分布，观察异常值
	# 查看日期
	regdate = data['regDate'].tolist()		# 19910001 20151212		异常
	creatdate = data['creatDate'].tolist()	# 20150618 20160407		需进一步查看
	#print(np.min(regdate),np.max(regdate))
	n = len(data['SaleID'])
	regyear,regmonth,regday = read_ymd(regdate)
	#print(regyear,regmonth,regday)
	'''
	for i in range(n):								# 仅Month报错‘0’
		if regyear[i] > 2015 or regyear[i] < 1991:
			print(regyear[i],'Year False')
		if regmonth[i] > 12 or regmonth[i] < 1:
			print(regmonth[i],'Month False')
		if regday[i] > 31 or regday[i] < 1:
			print(regday[i],'Day False')'''
	creatyear,creatmonth,creatday = read_ymd(creatdate)
	#print(creatyear,creatmonth,creatday)
	'''
	for i in range(n):								# 无报错
		if creatyear[i] > 2016 or creatyear[i] < 2015:
			print(creatyear[i],'Year False')
		if creatmonth[i] > 12 or creatmonth[i] < 1:
			print(creatmonth[i],'Month False')
		if creatday[i] > 31 or creatday[i] < 1:
			print(creatday[i],'Day False')'''
	n = len(test['SaleID'])
	regdate = test['regDate'].tolist()		# 19910001 20151211		异常
	creatdate = test['creatDate'].tolist()	# 20140310 20160407		需进一步查看
	#print(np.min(regdate),np.max(regdate))
	regyear,regmonth,regday = read_ymd(regdate)
	#print(regyear,regmonth,regday)
	'''
	for i in range(n):								# 仅Month报错‘0’
		if regyear[i] > 2015 or regyear[i] < 1991:
			print(regyear[i],'Year False')
		if regmonth[i] > 12 or regmonth[i] < 1:
			print(regmonth[i],'Month False')
		if regday[i] > 31 or regday[i] < 1:
			print(regday[i],'Day False')'''
	creatyear,creatmonth,creatday = read_ymd(creatdate)
	#print(creatyear,creatmonth,creatday)
	'''
	for i in range(n):								# 无报错
		if creatyear[i] > 2016 or creatyear[i] < 2014:
			print(creatyear[i],'Year False')
		if creatmonth[i] > 12 or creatmonth[i] < 1:
			print(creatmonth[i],'Month False')
		if creatday[i] > 31 or creatday[i] < 1:
			print(creatday[i],'Day False')'''

	# 查看二分类数据的值
	#print(data['gearbox'].value_counts())				# 0.0	111623		1.0		32396
	#print(data['notRepairedDamage'].value_counts())	# 0.0	111361		1.0		14315	-	24324
	#print(data['seller'].value_counts())				# 0		149999		1		1
	#print(data['offerType'].value_counts())			# 0		150000
	#print(test['gearbox'].value_counts())				# 0.0	37131		1.0		10901
	#print(test['notRepairedDamage'].value_counts())	# 0.0	37224		1.0		4707	-	8069
	#print(test['seller'].value_counts())				# 0		50000
	#print(test['offerType'].value_counts())			# 0		50000

	# 查看各特征数据是否有某个值占有绝大部分
	redundant = []
	for i in data.columns:
		counts = data[i].value_counts()
		count_max = counts.iloc[0]
		if count_max / len(data) * 100 > 99:
			redundant.append(i)
	redundant = list(redundant)
	#print(redundant)

	'''
	# 箱型图
	data.drop('SaleID',axis=1,inplace=True)
	test.drop('SaleID',axis=1,inplace=True)
	data.drop(['seller','offerType'],axis=1,inplace=True)
	test.drop(['seller','offerType'],axis=1,inplace=True)
	data_c = deepcopy(data)
	test_c = deepcopy(test)
	data_c.drop(['gearbox','notRepairedDamage','bodyType','fuelType'],axis=1,inplace=True)
	test_c.drop(['gearbox','notRepairedDamage','bodyType','fuelType'],axis=1,inplace=True)
	data_c['model'] = data_c['model'].fillna(sc.stats.mode(data_c['model'])[0][0])		# 众数填充model
	box_all(data_c)
	box_all(test_c)'''
	'''
	# 直方图
	data.drop('SaleID',axis=1,inplace=True)
	test.drop('SaleID',axis=1,inplace=True)
	data.drop(['seller','offerType'],axis=1,inplace=True)
	test.drop(['seller','offerType'],axis=1,inplace=True)
	data_c = deepcopy(data)
	test_c = deepcopy(test)
	data_c['model'] = data_c['model'].fillna(sc.stats.mode(data_c['model'])[0][0])		# 众数填充model
	data_c['fuelType'] = data_c['fuelType'].fillna(-1)
	data_c['gearbox'] = data_c['gearbox'].fillna(-1)
	data_c['bodyType'] = data_c['bodyType'].fillna(-1)
	data_c['notRepairedDamage'].replace('-',-1,inplace=True)
	data_c['notRepairedDamage'].replace('0.0',0,inplace=True)
	data_c['notRepairedDamage'].replace('1.0',1,inplace=True)
	#hist_all(data_c)
	test_c['fuelType'] = test_c['fuelType'].fillna(-1)
	test_c['gearbox'] = test_c['gearbox'].fillna(-1)
	test_c['bodyType'] = test_c['bodyType'].fillna(-1)
	test_c['notRepairedDamage'].replace('-',-1,inplace=True)
	test_c['notRepairedDamage'].replace('0.0',0,inplace=True)
	test_c['notRepairedDamage'].replace('1.0',1,inplace=True)
	hist_all(test_c)'''
	'''
	# 通过Z方法判断异常值 
	cols = ['kilometer','price','v_0','v_1','v_2','v_3','v_4','v_5','v_6','v_7','v_8','v_9','v_10','v_11','v_12','v_13','v_14']		# 获取列名
	data_out = data[['kilometer','price','v_0','v_1','v_2','v_3','v_4','v_5','v_6','v_7','v_8','v_9','v_10','v_11','v_12','v_13','v_14']]	# 获取需要的列
	outliers = get_outliers(data_out,cols)
	for i in range(len(cols)):
		print(len(outliers[i]))				# 每列异常值个数
					# 1840 2880 3601 0 4506 242 446 4506 0 4506 28 343 217 4506 895 264 2246
	cols = ['kilometer','v_0','v_1','v_2','v_3','v_4','v_5','v_6','v_7','v_8','v_9','v_10','v_11','v_12','v_13','v_14']		# 获取列名
	test_out = test[['kilometer','v_0','v_1','v_2','v_3','v_4','v_5','v_6','v_7','v_8','v_9','v_10','v_11','v_12','v_13','v_14']]	# 获取需要的列
	outliers = get_outliers(test_out,cols)
	for i in range(len(cols)):
		print(len(outliers[i]))				# 每列异常值个数
					# 601 1195 0 1504 67 134 1504 0 1504 14 113 61 1504 303 76 743
	# v2 v5 v7 v11 异常值都一模一样'''

	# v2 v5 v7 v11 关系散点图
	v_cols = ['v_2','v_5','v_7','v_11']
	#scatter_v(data,v_cols)		# 4者可能存在共线性
	'''
	# 价格分布图
	plt.figure(figsize=(10,6))
	sn.distplot(data['price'],hist_kws={"edgecolor": (1,0,0,1)})
	plt.show()
	print("Skewness: %f" % data['price'].skew())	# 右偏长尾	Skewness: 3.346466
	print("Kurtosis: %f" % data['price'].kurt())	# 陡峭		Kurtosis: 18.998174	'''
	'''
	price = data['price'].tolist()
	s0 = 0
	s1 = 0
	s2 = 0
	s3 = 0
	s4 = 0
	for i in range(n):
		if price[i] < 1000:
			s0 = s0+1
		elif price[i] >= 1000 and price[i] < 5000:
			s1 = s1+1
		elif price[i] >= 5000 and price[i] < 10000:
			s2 = s2+1
		elif price[i] >= 10000 and price[i] < 20000:
			s3 = s3+1
		else:
			s4 = s4+1
	s0 = s0/n
	s1 = s1/n
	s2 = s2/n
	s3 = s3/n
	s4 = s4/n
	print(s0,s1,s2,s3,s4)	'''
	# <1000		1000-5000	5000-10000	10000-20000		>20000
	# 0.18984	0.44018666	0.1935		0.12798		0.048493333



	#################################### 一起清洗train、test数据，合并处理
	# 备份
	data_c = deepcopy(data)
	test_c = deepcopy(test)

	# 去除train异常点 (v3<-6、v4>5、v13>10、v14>5、model缺失)
	data = data[data.v_3 > -6.0]
	data.reset_index(drop = True, inplace = True)
	#print(data.shape)

	#qt = QuantileTransformer(output_distribution='normal', random_state=0)
	#data['price'] = qt.fit_transform(data['price'].reshape(-1, 1))
	#plt.figure(figsize=(10,6))
	#sn.distplot(data['price'],hist_kws={"edgecolor": (1,0,0,1)})
	#plt.show()

	# boxcox(price) 使价格近似正太分布
	lam_range = np.linspace(0,1,100)
	llf = np.zeros(lam_range.shape, dtype=float)
	for i,lam in enumerate(lam_range):			# lambda 估算似然函数
		llf[i] = sc.stats.boxcox_llf(lam, data['price'])		# y 必须>0
	lam_best = lam_range[llf.argmax()]
	data['price'] = sc.special.boxcox1p(data['price'], lam_best)
	#print(lam_best)
	'''
	plt.figure(figsize=(10,6))
	sn.distplot(data['price'],hist_kws={"edgecolor": (1,0,0,1)})
	plt.show()
	print("Skewness: %f" % data['price'].skew())	# Skewness: -0.012828
	print("Kurtosis: %f" % data['price'].kurt())	# Kurtosis: -0.369006
	print(sc.stats.anderson(data['price']))'''
	# AndersonResult(statistic=138.46198330871994, critical_values=array([0.576, 0.656, 0.787, 0.918, 1.092]), 
	#significance_level=array([15. , 10. ,  5. ,  2.5,  1. ]))
	# 如果输出的统计量值statistic < critical_values，则表示在相应的significance_level下，接受原假设，
	# 认为样本数据来自给定的正态分布。

	# 合并train、test特征数据为 Merge_feat
	train_feat = data.drop('price',axis=1)
	Merge_feat = pd.concat([train_feat,test]).reset_index(drop=True)
	#Merge_feat.info()

	# 去除不需要的列
	Merge_feat.drop(['SaleID','seller','offerType'], axis=1, inplace=True)

	# 修改notRepairedDamage为int值 0，1，-1
	Merge_feat['notRepairedDamage'].replace('-',-1,inplace=True)
	Merge_feat['notRepairedDamage'].replace('0.0',0,inplace=True)
	Merge_feat['notRepairedDamage'].replace('1.0',1,inplace=True)
	
	# 设置bodyType、fuelType、gearbox 缺失值为-1
	Merge_feat['fuelType'] = Merge_feat['fuelType'].fillna(-1)
	Merge_feat['gearbox'] = Merge_feat['gearbox'].fillna(-1)
	Merge_feat['bodyType'] = Merge_feat['bodyType'].fillna(-1)

	# 替换月份0，保留年月
	regdate = Merge_feat['regDate'].tolist()
	regyear,regmonth,regday = read_ymd(regdate)
	regmonth = regmonth_replace0(Merge_feat)
	regdate = list((np.array(regyear)*100) + np.array(regmonth))	# 保留年月作为regDate
	Merge_feat['regDate'] = regdate
	#print(Merge_feat['regDate'])

	# 汽车注册日期与上线日期之年差
	creatdate = Merge_feat['creatDate'].tolist()
	creatyear,creatmonth,creatday = read_ymd(creatdate)
	creatdate = list((np.array(creatyear)*100) + np.array(creatmonth))	# 保留年月作为creatDate
	usedyear = get_usedyear(regdate,creatdate)
	# 取代注册日期
	Merge_feat['usedyear'] = usedyear
	Merge_feat.drop(['regDate'],axis=1,inplace=True)

	# 所有数据保留3位小数
	#Merge_feat = Merge_feat.round(3)
	#lookinto_data(Merge_feat)

	# 备份，导出到data和test
	Merge_feat_c = deepcopy(Merge_feat)
	n_merge = len(Merge_feat['model'])
	n_train = len(data['SaleID'])
	n_test = len(test['SaleID'])
	x = Merge_feat[0:n_train]
	y = data['price']
	data = pd.concat([x,y],axis=1).reset_index(drop=True)
	test = Merge_feat[n_train:n_merge]
	test.reset_index(drop=True)

	# 输出清洗后数据
	data.to_csv('train_clean.csv', index=True, header=True, encoding='utf_8') # Chinese 'utf_8_sig'
	test.to_csv('test_clean.csv', index=True, header=True, encoding='utf_8') # Chinese 'utf_8_sig'
	# 打印出将要输出文件中的前五行记录
	#print(data[:5].to_csv())



	######################################### 观察train各个变量与价格间的关系，以及其之间的共线性关系
	# 各个连续变量与价格散点图
	cols = data.columns
	sep_cols = ['bodyType','fuelType','gearbox','notRepairedDamage']
	con_cols = [x for x in cols if x not in sep_cols]
	#scatter_price(data,con_cols[0:3])
	#scatter_price(data,con_cols[3:6])
	#scatter_price(data,con_cols[6:9])
	#scatter_price(data,con_cols[9:12])
	#scatter_price(data,con_cols[12:15])
	#scatter_price(data,con_cols[15:18])
	#scatter_price(data,con_cols[18:21])
	#scatter_price(data,con_cols[21:24])

	# 各个离散变量与价格箱型图、小提琴图
	#box_price(data,sep_cols)
	#violin_price(data,sep_cols)

	# 各个特征间多重共线性vif检验
	#vif = get_vif(Merge_feat)
	#print(vif)								# 无10以上vif值，v1、v10较高
	'''
	             VIF            feature
	0   1.699025e+00               name
	1   1.926377e+00              model
	2   1.284742e+00              brand
	3   1.430551e+00           bodyType
	4   1.278963e+00           fuelType
	5   1.235058e+00            gearbox
	6   1.072410e+00              power
	7   1.499489e+00          kilometer
	8   1.060606e+00  notRepairedDamage
	9   1.092444e+00         regionCode
	10  1.000415e+00          creatDate
	11  3.473187e+06                v_0
	12  7.834521e+07                v_1
	13  4.770966e+07                v_2
	14  3.747069e+06                v_3
	15  1.213635e+05                v_4
	16  1.733289e+04                v_5
	17  2.368670e+04                v_6
	18  1.390119e+04                v_7
	19  1.445108e+03                v_8
	20  1.006291e+03                v_9
	21  7.714300e+07               v_10
	22  4.649012e+07               v_11
	23  1.989772e+07               v_12
	24  2.868261e+05               v_13
	25  5.935834e+02               v_14
	26  6.669951e+00           usedyear'''
	#vif_cartype = get_vif(Merge_feat[['name','model','brand','bodyType','fuelType','gearbox','power']])
	#print(vif_cartype)
	'''
	0  1.002820      name
	1  1.189768     model
	2  1.163911     brand
	3  1.085276  bodyType
	4  1.075228  fuelType
	5  1.079994   gearbox
	6  1.037132     power'''
	#vif_used = get_vif(Merge_feat[['kilometer','notRepairedDamage','usedyear']])
	#print(vif_used)
	'''
	0  1.317346          kilometer
	1  1.003114  notRepairedDamage
	2  1.320676           usedyear'''
	#vif_v = get_vif(Merge_feat[['v_0','v_1','v_2','v_3','v_4','v_5','v_6','v_7','v_8','v_9','v_10','v_11','v_12','v_13','v_14']])
	#print(vif_v)								# v9 vif较高
	'''
	0   3.230462e+06     v_0
	1   7.834099e+07     v_1
	2   4.770089e+07     v_2
	3   3.415155e+06     v_3
	4   1.065668e+05     v_4
	5   1.709006e+04     v_5
	6   2.298579e+04     v_6
	7   1.364491e+04     v_7
	8   1.425862e+03     v_8
	9   9.822703e+02     v_9
	10  7.693606e+07    v_10
	11  4.613166e+07    v_11
	12  1.949494e+07    v_12
	13  2.508672e+05    v_13
	14  5.268167e+02    v_14'''
	#vif_v257 = get_vif(Merge_feat[['v_2','v_5','v_7','v_11']])
	#print(vif_v257)								# v2, v5, v7 均有严重共线性
	'''
	0  19.497803     v_2
	1  10.292323     v_5
	2  24.874440     v_7
	3   3.554847    v_11'''

	# 各个特征间多重共线性spearman相关系数检验
	#get_correlation(Merge_feat[['v_2','v_5','v_7','v_11']])
	#get_correlation(Merge_feat)
	'''
	|r|>0.8:
		v1-v6v10, v3-v8v12usedyear, v4-v9v13, v6-v10, v8-v12, v9-v13
	|r|>0.9:
		v3-v8, v4-v9v13, v6-v10, v8-v12'''

	# GBDT查看各特征重要性，去除低重要性特征
	GBR_best = GradientBoostingRegressor(loss='huber', alpha=0.8, learning_rate=0.1, n_estimators=300, subsample=0.9,
		min_samples_split=100, min_samples_leaf=100, max_depth=5, max_features='sqrt', warm_start=True)
	GBR_best.fit(x,y.ravel())
	y_predict = GBR_best.predict(x)
	# MAE
	MAE = mean_absolute_error(y, y_predict)
	#print(MAE)	# 0.24621257092907017
	fi = GBR_best.feature_importances_
	#print(GBR_best.feature_importances_)
	'''[1.05527997e-03 3.37936565e-04 7.29032416e-04 4.63131802e-04
	 6.24771253e-05 4.51477052e-04 2.91927948e-02 1.86100098e-02
	 5.29207264e-03 3.65525662e-05 2.34776032e-05 2.13330920e-01
	 3.52940072e-03 1.23078562e-02 1.65955597e-01 1.34391887e-03
	 1.21355687e-02 1.51494427e-02 2.69971851e-03 6.26575779e-02
	 2.14799364e-03 6.10062310e-02 1.34141227e-02 2.64600819e-01
	 7.14602275e-04 3.77171765e-03 1.08980270e-01]'''



	######################################### 特征标准化
	# 查看特征的偏度、峰度
	skew_feat = Merge_feat.apply(lambda x: sc.stats.skew(x)).sort_values(ascending=False)
	high_skew = skew_feat[skew_feat > 0.5]
	kurt_feat = Merge_feat.apply(lambda x: sc.stats.kurtosis(x)).sort_values(ascending=False)
	high_kurt = kurt_feat[kurt_feat > 0.5]
	#print(high_skew,high_kurt)
	'''						skew
	power                64.316444
	v_7                   5.129350
	v_2                   4.842483
	v_11                  3.025913
	model                 1.485134
	brand                 1.150620
	bodyType              0.930870
	fuelType              0.798889
	regionCode            0.688620
	name                  0.557136
	gearbox               0.486540
	v_9                   0.419966
	v_6                   0.371045
	v_4                   0.369368
	v_12                  0.369296
	v_1                   0.362463
	v_13                  0.263009
	v_8                   0.208077
	v_3                   0.101283
	v_10                  0.023466
	usedyear             -0.023844
	notRepairedDamage    -0.125175
	v_14                 -1.195883
	v_0                  -1.314828
	kilometer            -1.525291
	v_5                  -4.735072
	creatDate           -95.310159
							kurt
	creatDate            11375.041375
	power                 5383.101950
	v_7                     25.838936
	v_2                     23.857117
	v_5                     22.916957
	v_11                    12.546198
	v_0                      3.988952
	fuelType                 3.870337
	v_14                     2.373723
	model                    1.739019
	kilometer                1.139672
	brand                    1.077587
	notRepairedDamage        0.855690
	gearbox                  0.484283
	v_12                     0.281394
	bodyType                 0.191763
	v_4                     -0.206617
	v_9                     -0.323975
	regionCode              -0.342743
	v_3                     -0.424462
	v_13                    -0.476954
	v_10                    -0.574167
	v_8                     -0.633266
	usedyear                -0.701787
	name                    -1.038320
	v_6                     -1.739570
	v_1                     -1.750152'''
	'''
	# 查看各特征分布
	cols = Merge_feat.columns
	fig,axes = plt.subplots(1,3,figsize=(10,10))
	fig.subplots_adjust(hspace=0.5, wspace=0.5)
	for i,ax in zip(cols[24:27],axes.flatten()):
		Merge_feat[i].hist(bins=30, ax=ax)
		ax.set_title(str(i),size=8)
	plt.show()
	#print(Merge_feat['kilometer'].value_counts())	'''

	# 去除低重要性特征
	Merge_feat.drop(['regionCode','creatDate'], axis=1, inplace=True)	# 去除低重要性特征：regionCode，creatDate
	#print(Merge_feat.columns)

	cols = Merge_feat.columns
	sep_cols = ['bodyType','fuelType','gearbox','notRepairedDamage','kilometer']
	con_cols = [x for x in cols if x not in sep_cols]
	v_cols = ['v_0','v_1','v_2','v_3','v_4','v_5','v_6','v_7','v_8','v_9','v_10','v_11','v_12','v_13','v_14']

	# 连续数据标准化
	''' boxcox
	lam_range = np.linspace(0,1,100)
	llf = np.zeros(lam_range.shape, dtype=float)
	for j in con_cols:
		for i,lam in enumerate(lam_range):			# lambda 估算似然函数
			llf[i] = sc.stats.boxcox_llf(lam, Merge_feat[j])		# y 必须>0
		lam_best = lam_range[llf.argmax()]
		Merge_feat[j] = sc.special.boxcox1p(Merge_feat[j], lam_best)'''

	# 连续数据标准化				# model,brand,power 0值过多
	# quantile
	qt = QuantileTransformer(output_distribution='normal', random_state=0)
	for i in con_cols:
		Merge_feat[i] = qt.fit_transform(Merge_feat[i].values.reshape(-1, 1))
	# PCA降维v系列
	X = Merge_feat[v_cols]
	pca = PCA(n_components=7)		# 降到7维 (copy=True, n_components=7, whiten=False)
	pca.fit(X)						# 训练
	newX = pca.fit_transform(X)		# 降维后的数据
	#print(pca.explained_variance_ratio_)	# 贡献率
	# [0.43158907 0.18529884 0.13846617 0.13140018 0.04131971 0.03479909 0.02117432]
	#print(X.shape,newX.shape)
	x_cols = ['x_'+str(i) for i in range(7)]
	Merge_v = pd.DataFrame(newX, index=None, columns=x_cols)
	Merge_feat = pd.concat([Merge_feat,Merge_v],axis=1)
	Merge_feat = Merge_feat.drop(v_cols,axis=1)
	#print(Merge_feat.columns)

	# 离散数据dummy
	# dummy:	body,fuel,gear,notrepairdamage,km
	Merge_cat = Merge_feat[sep_cols]
	Merge_cat = Merge_cat.astype(str)
	Merge_cat = pd.get_dummies(Merge_cat)
	#print(Merge_cat.columns,Merge_cat.shape)
	Merge_feat = pd.concat([Merge_feat,Merge_cat],axis=1)
	Merge_feat = Merge_feat.drop(sep_cols,axis=1)
	print(Merge_feat.columns,Merge_feat.shape)

	# 备份，导出到data和test
	Merge_feat_c = deepcopy(Merge_feat)
	x = Merge_feat[0:n_train]
	y = data['price']
	data = pd.concat([x,y],axis=1).reset_index(drop=True)
	test = Merge_feat[n_train:n_merge]
	test.reset_index(drop=True)
	print(data.shape,test.shape)

	# 输出清洗后数据
	data.to_csv('train_clean_new.csv', index=False, header=True, encoding='utf_8') # Chinese 'utf_8_sig'
	test.to_csv('test_clean_new.csv', index=False, header=True, encoding='utf_8') # Chinese 'utf_8_sig'
	# 打印出将要输出文件中的前五行记录
	#print(data[:5].to_csv())


