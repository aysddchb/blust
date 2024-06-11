# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

### 假设数据集有'flight_number', 'date', 'departure_airport', 'arrival_airport', 'scheduled_departure_time', 'actual_departure_time', 'scheduled_arrival_time', 'actual_arrival_time', 'weather', 'airline'，其中'scheduled_departure_time'和'scheduled_arrival_time
# 从CSV文件加载航班数据
flights_data = pd.read_csv(r'D:\python\兼职\4.17\flights.csv')
# 删除不需要的列
flights_data = flights_data.drop(['flight_number', 'date', 'actual_departure_time', 'actual_arrival_time'], axis=1)

# 处理缺失值
flights_data = flights_data.dropna()

# 对类别型特征进行独热编码
flights_data = pd.get_dummies(flights_data, columns=['departure_airport', 'arrival_airport', 'weather', 'airline'])

# 将时间戳转换为小时
flights_data['scheduled_departure_hour'] = pd.to_datetime(flights_data['scheduled_departure_time']).dt.hour
flights_data['scheduled_arrival_hour'] = pd.to_datetime(flights_data['scheduled_arrival_time']).dt.hour

# 删除原始时间戳列
flights_data = flights_data.drop(['scheduled_departure_time', 'scheduled_arrival_time'], axis=1)

# 将数据集拆分为特征集（X）和目标变量（y），然后将其划分为训练集和测试集
X = flights_data.drop('on_time', axis=1)
y = flights_data['on_time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 创建随机森林分类器
rf_model = RandomForestClassifier(random_state=42)

# 使用网格搜索交叉验证寻找最佳参数组合
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 输出最佳参数组合
print("最佳参数组合:", grid_search.best_params_)

# 使用最佳参数组合的模型进行预测
best_rf_model = grid_search.best_estimator_

# 预测
y_pred = best_rf_model.predict(X_test)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
print("模型准确率:", accuracy)

# 计算精确率、召回率和F1值
print(classification_report(y_test, y_pred))

# 输出混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print("混淆矩阵:\n", conf_matrix)

##### 深入分析

# 性能对比
# 将随机森林模型与逻辑回归和支持向量机进行对比，并评估它们在航班正点率预测任务上的性能优劣
# 初始化逻辑回归模型
logistic_model = LogisticRegression()

# 训练逻辑回归模型
logistic_model.fit(X_train, y_train)

# 预测并评估逻辑回归模型性能
y_pred_logistic = logistic_model.predict(X_test)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print("逻辑回归模型准确率:", accuracy_logistic)
print(classification_report(y_test, y_pred_logistic))

# 初始化支持向量机模型
svm_model = SVC()

# 训练支持向量机模型
svm_model.fit(X_train, y_train)

# 预测并评估支持向量机模型性能
y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("支持向量机模型准确率:", accuracy_svm)
print(classification_report(y_test, y_pred_svm))

# 特征重要性分析
# 利用随机森林模型提供的特征重要性信息，分析哪些特征对航班正点率的预测具有重要影响
# 输出特征重要性
feature_importances = rf_model.feature_importances_
print("特征重要性:")
for feature, importance in zip(X.columns, feature_importances):
    print(f"{feature}: {importance}")

# 误差分析
# 对模型预测错误的案例进行深入剖析，探究导致预测错误的可能原因
# 找出预测错误的样本
incorrect_predictions = X_test[y_test != y_pred]

# 输出预测错误的样本
print("预测错误的样本:")
print(incorrect_predictions)


# 模型稳定性分析
# 通过多次实验和交叉验证，评估模型的稳定性

# 执行交叉验证
cv_scores = cross_val_score(rf_model, X, y, cv=5)
print("交叉验证得分:", cv_scores)

##### 添加图表

# 特征重要性图表
feature_importances = best_rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.show()

# 混淆矩阵热图
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 学习曲线
train_sizes, train_scores, test_scores = learning_curve(best_rf_model, X, y, cv=3, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score', color='blue')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.1)
plt.plot(train_sizes, test_mean, label='Cross-validation score', color='red')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='red', alpha=0.1)
plt.title('Learning Curve')
plt.xlabel('Number of training samples')
plt.ylabel('Score')
plt.legend(loc='best')
plt.show()

# ROC曲线和AUC值
y_score = best_rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# 误差分析图表
# 计算预测错误的概率
y_prob = best_rf_model.predict_proba(X_test)
error_prob = np.max(y_prob, axis=1)
# 将误差概率添加到测试集中
error_analysis_df = X_test.copy()
error_analysis_df['Error Probability'] = error_prob
# 将预测结果添加到测试集中
error_analysis_df['Predicted'] = y_pred
# 标记出预测错误的样本
error_analysis_df['Error'] = error_analysis_df['Predicted'] != y_test

# 绘制误差分析散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Error Probability', y='scheduled_departure_hour', hue='Error', data=error_analysis_df, palette={True: 'red', False: 'blue'})
plt.title('Error Analysis: Probability vs Scheduled Departure Hour')
plt.xlabel('Error Probability')
plt.ylabel('Scheduled Departure Hour')
plt.show()


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# 生成模拟的航班数据集
def generate_flight_data(num_flights):
    # 创建随机种子，确保结果可复现
    np.random.seed(0)

    # 生成航班数据
    flights_data = pd.DataFrame()
    flights_data['scheduled_departure_time'] = pd.date_range(start='2022-01-01 00:00:00', periods=num_flights, freq='H')
    flights_data['scheduled_arrival_time'] = flights_data['scheduled_departure_time'] + pd.to_timedelta(np.random.randint(1, 6, size=num_flights), unit='m')
    flights_data['departure_airport'] = np.random.choice(['JFK', 'LAX', 'ORD', 'DFW'], size=num_flights)
    flights_data['arrival_airport'] = np.random.choice(['LHR', 'CDG', 'HND', 'PEK'], size=num_flights)
    flights_data['weather'] = np.random.choice(['sunny', 'cloudy', 'rainy'], size=num_flights)
    flights_data['airline'] = np.random.choice(['AA', 'DL', 'UA', 'WN'], size=num_flights)

    # 生成温度、能见度、风速、风向数据
    flights_data['temperature'] = np.random.randint(-10, 30, size=num_flights)  # 温度范围：-10°C 到 30°C
    flights_data['visibility'] = np.random.randint(200, 2000, size=num_flights)  # 能见度范围：200m 到 2000m
    flights_data['wind_speed'] = np.random.uniform(0, 25, size=num_flights)  # 风速范围：0米/秒 到 25米/秒
    flights_data['wind_direction'] = np.random.choice(['headwind', 'tailwind', 'crosswind'], size=num_flights)
    flights_data['aircraft_model'] = np.random.choice(['B747','B737','B777','B787','B767','B757','A320','A350','A380','A330','A319','A321'], size=num_flights)  # 机型

    # 根据风向调整风速对飞行的影响
    for idx, row in flights_data.iterrows():
        if row['wind_direction'] == 'headwind':
            # 逆风，对飞机影响较大，增加风速
            flights_data.at[idx, 'wind_speed'] += np.random.uniform(5, 15)
        elif row['wind_direction'] == 'tailwind':
            # 顺风，对飞机有帮助，减少风速
            flights_data.at[idx, 'wind_speed'] -= np.random.uniform(0, 5)
        else:
            # 侧风，对飞机影响较大，增加风速
            flights_data.at[idx, 'wind_speed'] += np.random.uniform(10, 20)
    
    # 根据天气和风速等因素判断航班是否正点
    flights_data['on_time'] = True
    for idx, row in flights_data.iterrows():
        if row['weather'] == 'rainy' or row['visibility'] < 600 or row['wind_speed'] > 20:
            # 下雨、能见度低于600m、风速大于20米/秒，航班可能延误
            flights_data.at[idx, 'on_time'] = False
            
    # 根据条件生成飞机是否正点的标签
    def generate_on_time(weather, temperature, visibility, wind_speed, wind_direction, aircraft_model):
        # 根据实际情况定义飞机起飞的条件
        if (weather == 'sunny' and visibility >= 600 and wind_speed <= 20) or \
           (weather == 'cloudy' and visibility >= 400 and wind_speed <= 20) or \
           (weather == 'rainy' and visibility >= 400 and wind_speed <= 20):
            return random.choices([True, False],weights=[0.7,0.3])[0]
        else:
            return False

    flights_data['on_time'] = flights_data.apply(lambda row: generate_on_time(row['weather'], row['temperature'], row['visibility'], row['wind_speed'], row['wind_direction'], row['aircraft_model']), axis=1)

    # 生成机型数据

    return flights_data

# 生成示例数据集
flights_data = generate_flight_data(10000)

# 将数据集保存为CSV文件
flights_data.to_csv(r'D:\python\兼职\4.17\flights.csv', index=False)



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer
# 加载航班数据
flights_data = pd.read_csv(r'D:\python\兼职\4.17\flights.csv')

# 删除不需要的列
flights_data = flights_data.drop(['flight_number', 'date', 'actual_departure_time', 'actual_arrival_time'], axis=1)

# 处理缺失值
flights_data = flights_data.dropna()

# 将类别型特征进行独热编码
flights_data = pd.get_dummies(flights_data, columns=['departure_airport', 'arrival_airport', 'airline'])
flights_data.columns
# 将时间戳转换成时间间隔（以分钟为单位）
flights_data['scheduled_departure_time'] = pd.to_datetime(flights_data['scheduled_departure_time'])
flights_data['scheduled_arrival_time'] = pd.to_datetime(flights_data['scheduled_arrival_time'])

# 计算距离某一参考时间点的时间间隔（单位：分钟）
reference_time = pd.to_datetime('2022-01-01')  # 参考时间点，可以是数据集中的最小时间
flights_data['departure_time_delta'] = (flights_data['scheduled_departure_time'] - reference_time).dt.total_seconds() / 60
flights_data['arrival_time_delta'] = (flights_data['scheduled_arrival_time'] - reference_time).dt.total_seconds() / 60

# 删除原始时间戳列
flights_data = flights_data.drop(['scheduled_departure_time', 'scheduled_arrival_time'], axis=1)

# 添加新的特征工程，考虑天气、温度、能见度、风向风速以及机型的影响
# 根据条件进行标记
# 使用标签编码将分类特征转换为数值型特征
label_encoder = LabelEncoder()
flights_data['weather'] = label_encoder.fit_transform(flights_data['weather'])
flights_data['departure_airport_DFW'] = label_encoder.fit_transform(flights_data['departure_airport_DFW'])
flights_data['departure_airport_JFK'] = label_encoder.fit_transform(flights_data['departure_airport_JFK'])
flights_data['departure_airport_LAX'] = label_encoder.fit_transform(flights_data['departure_airport_LAX'])
flights_data['departure_airport_ORD'] = label_encoder.fit_transform(flights_data['departure_airport_ORD'])
flights_data['arrival_airport_CDG'] = label_encoder.fit_transform(flights_data['arrival_airport_CDG'])
flights_data['arrival_airport_HND'] = label_encoder.fit_transform(flights_data['arrival_airport_HND'])
flights_data['arrival_airport_LHR'] = label_encoder.fit_transform(flights_data['arrival_airport_LHR'])
flights_data['arrival_airport_PEK'] = label_encoder.fit_transform(flights_data['arrival_airport_PEK'])
flights_data['airline_AA'] = label_encoder.fit_transform(flights_data['airline_AA'])
flights_data['airline_DL'] = label_encoder.fit_transform(flights_data['airline_DL'])
flights_data['airline_UA'] = label_encoder.fit_transform(flights_data['airline_UA'])
flights_data['airline_WN'] = label_encoder.fit_transform(flights_data['airline_WN'])
flights_data['aircraft_model'] = label_encoder.fit_transform(flights_data['aircraft_model'])
# 将风向特征转换为数值型特征
flights_data['wind_direction'] = flights_data['wind_direction'].map({'headwind': 1, 'tailwind': -1, 'crosswind': 0})

# 将数据集拆分为特征集（X）和目标变量（y），然后将其划分为训练集和测试集
X = flights_data.drop('on_time', axis=1)
y = flights_data['on_time']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 定义参数网格
param_grid = {
    'n_estimators': [100, 200, 300, 400],  # 增加更多的树的数量，以进一步提高模型性能
    'max_depth': [None, 10, 20, 30, 50],  # 增加更深的树的最大深度，以适应更复杂的数据模式
    'min_samples_split': [2, 5, 10, 20, 30],  # 调整更多不同的最小样本拆分数，以优化模型的分裂过程
    'min_samples_leaf': [1, 2, 4, 8, 10],  # 调整更多不同的最小叶子节点样本数，以改善模型的叶子节点数量
    'max_features': ['auto', 'sqrt', 'log2']  # 考虑不同的特征数量来拆分节点，以提高模型的多样性
}

# 创建随机森林分类器
rf_model = RandomForestClassifier(random_state=42)

# 使用网格搜索交叉验证寻找最佳参数组合
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# 输出最佳参数组合
print("最佳参数组合:", grid_search.best_params_)

# 使用最佳参数组合的模型进行预测
best_rf_model = grid_search.best_estimator_

# 预测
y_pred = best_rf_model.predict(X_test)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
print("模型准确率:", accuracy)

# 计算精确率、召回率和F1值
print(classification_report(y_test, y_pred))

# 输出混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
print("混淆矩阵:\n", conf_matrix)

##### 深入分析

# 性能对比
# 初始化逻辑回归模型
logistic_model = LogisticRegression()

# 训练逻辑回归模型
logistic_model.fit(X_train, y_train)

# 预测并评估逻辑回归模型性能
y_pred_logistic = logistic_model.predict(X_test)
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print("逻辑回归模型准确率:", accuracy_logistic)
print(classification_report(y_test, y_pred_logistic))

# 初始化支持向量机模型
svm_model = SVC()

# 训练支持向量机模型
svm_model.fit(X_train, y_train)

# 预测并评估支持向量机模型性能
y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("支持向量机模型准确率:", accuracy_svm)
print(classification_report(y_test, y_pred_svm))

# 特征重要性分析
# 输出特征重要性
feature_importances = best_rf_model.feature_importances_
print("特征重要性:")
for feature, importance in zip(X.columns, feature_importances):
    print(f"{feature}: {importance}")

# 误差分析
# 找出预测错误的样本
incorrect_predictions = X_test[y_test != y_pred]

# 输出预测错误的样本
print("预测错误的样本:")
print(incorrect_predictions)

# 模型稳定性分析
# 执行交叉验证
cv_scores = cross_val_score(best_rf_model, X, y, cv=5)
print("交叉验证得分:", cv_scores)

##### 添加图表

# 特征重要性图表
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.show()

# 混淆矩阵热图
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 学习曲线
train_sizes, train_scores, test_scores = learning_curve(best_rf_model, X, y, cv=3, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training score', color='blue')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.1)
plt.plot(train_sizes, test_mean, label='Cross-validation score', color='red')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='red', alpha=0.1)
plt.title('Learning Curve')
plt.xlabel('Number of training samples')
plt.ylabel('Score')
plt.legend(loc='best')
plt.show()

# ROC曲线和AUC值
y_score = best_rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# 误差分析图表
# 计算预测错误的概率
y_prob = best_rf_model.predict_proba(X_test)
error_prob = np.max(y_prob, axis=1)
# 将误差概率添加到测试集中
error_analysis_df = X_test.copy()
error_analysis_df['Error Probability'] = error_prob
# 将预测结果添加到测试集中
error_analysis_df['Predicted'] = y_pred
# 标记出预测错误的样本
error_analysis_df['Error'] = error_analysis_df['Predicted'] != y_test

# 绘制误差分析散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Error Probability', y='departure_time_delta', hue='Error', data=error_analysis_df, palette={True: 'red', False: 'blue'})
plt.title('Error Analysis: Probability vs Scheduled Departure Hour')
plt.xlabel('Error Probability')
plt.ylabel('Scheduled Departure Hour')
plt.show()






