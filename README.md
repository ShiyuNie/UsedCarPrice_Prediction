# UsedCarPrice_Prediction
    利用机器学习算法预测二手车价格
    最佳分数566.48，排名452

## 项目介绍
### 背景：
				阿里云天池大赛：
				赛题以预测二手车的交易价格为任务，该数据来自某交易平台的二手车交易记录。
				主要用来练习python进行数据挖掘、机器学习等。
### 目的：
				预测二手车的交易价格
				评判标准 ： MAE(Mean Absolute Error)尽量小，则准确。
### 数据收集：
					来源：https://tianchi.aliyun.com/competition/entrance/231784/information
        总数据量超过40w，包含31列变量信息，其中15列为匿名变量。
        为了保证比赛的公平性，将会从中抽取15万条作为训练集，5万条作为测试集A，5万条作为测试集B，同时会对name、model、brand和regionCode等信息进行脱敏。
					字段表	    含义
					SaleID	  交易ID，唯一编码
					name	    汽车交易名称，已脱敏
					regDate	  汽车注册日期，例如20160101，2016年01月01日
					model	    车型编码，已脱敏
					brand	    汽车品牌，已脱敏
					bodyType	车身类型：豪华轿车：0，微型车：1，厢型车：2，大巴车：3，敞篷车：4，双门汽车：5，商务车：6，搅拌车：7
					fuelType	燃油类型：汽油：0，柴油：1，液化石油气：2，天然气：3，混合动力：4，其他：5，电动：6
					gearbox	  变速箱：手动：0，自动：1
					power	    发动机功率：范围 [ 0, 600 ]
					kilometer	汽车已行驶公里，单位万km
					notRepairedDamage	  汽车有尚未修复的损坏：是：0，否：1
					regionCode地区编码，已脱敏
					seller	  销售方：个体：0，非个体：1
					offerType	报价类型：提供：0，请求：1
					creatDate	汽车上线时间，即开始售卖时间
					price	    二手车交易价格（预测目标）
					v系列特征	 匿名特征，包含v0-14在内15个匿名特征
### 思路概要：
        * 观察数据
        * 清洗数据
        * 统计性分析
        * 用GBDT得到特征重要性，选择特征
        * 对匿名特征进行PCA降维
        * 特征标准化：
				连续型特征quantile转换为近似正太分布
				离散型特征dummy
        * 训练数据集划分 7:3，数据归一化
        * 使用线性回归（岭回归、Lasso回归、弹性网络回归）、支持向量机、GBDT、极端树、随机森林、XGBT、LGBM、Stacking、Blending建模和预测，比较所有模型的MAE分数，选择最优模型进行预测
        * 提交预测结果
        
## 文件说明
---
    UCRecord.txt                        数据分析记录文件
    /data/*.csv                         大赛提供的原始数据
    /user_data/*.csv                    清洗后数据
    data_prepro.py                      数据预处理、清洗
    /figures/*.png,*.pdf                数据分布图、相关性图等
    data_regression.py                  建模、调参、预测
    scores_of_models.png		各模型MAE分数（数据集train:test=7:3）
    submission.csv                      预测结果文件
 
		
		
