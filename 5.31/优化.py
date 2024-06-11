# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 23:37:03 2024

@author: 86182
"""

import random
import numpy as np
import pandas as pd
import math

# 生成航班数据集后，假设数据已经准备好，我们可以直接使用
# flights 是一个包含航班信息的 DataFrame

flights = pd.read_csv("flights.csv")  # 如果有航班数据集的 CSV 文件
aircrafts = pd.read_csv('aircrafts.csv')
airports = pd.read_csv('airports.csv')

# 初始化信息素
num_flights = len(flights)
pheromones = np.ones((num_flights, num_flights))  # 用矩阵来表示航班之间的信息素

# 约束条件
aircraft_available_time = aircrafts['Available_Time'].tolist()
airport_capacity = airports['Capacity'].tolist()

# 定义模拟退火参数
initial_temperature = 1000  # 初始温度
cooling_rate = 0.95  # 冷却速率
min_temperature = 1e-3  # 最低温度
num_iterations = 100  # 每个温度的迭代次数

# 初始化参数范围和初始值
best_parameters = {'num_ants': 10, 'evaporation_rate': 0.1, 'alpha': 1.0, 'beta': 2.0}
current_parameters = best_parameters.copy()
current_cost = float('inf')

def evaluate_parameters(parameters):
    # 初始化信息素
    pheromones = np.ones((num_flights, num_flights))
    
    # 初始化航班延误情况
    flights['Delayed'] = False
    flights['Delay'] = 0
    
    # 计算航班延误并更新信息素
    for _ in range(num_iterations):
        for ant in range(int(parameters['num_ants'])):
            selected_flight = None
            max_pheromone = -1

            for idx, flight in flights.iterrows():
                pheromone = pheromones[ant][idx]
                if flight['Delayed']:
                    delay = random.randint(0, 10)
                else:
                    delay = 0
                
                # 考虑航班延误、信息素和延误时间的综合因素
                score = pheromone / (1 + delay)
                if score > max_pheromone:
                    selected_flight = idx
                    max_pheromone = score

            # 更新航班延误
            if selected_flight is not None:
                flights.at[selected_flight, 'Delayed'] = True
                flights.at[selected_flight, 'Delay'] = delay

        # 更新信息素
        for idx, flight in flights.iterrows():
            for ant in range(parameters['num_ants']):
                pheromones[ant][idx] *= (1 - parameters['evaporation_rate'])

                # 根据航班延误程度更新信息素
                if flight['Delayed']:
                    pheromones[ant][idx] += parameters['alpha'] * (1 / (1 + flight['Delay']))
                else:
                    pheromones[ant][idx] += parameters['beta']

    # 计算总延误信息
    total_delay = flights['Delay'].sum()
    return total_delay

# 模拟退火算法
temperature = initial_temperature
while temperature > min_temperature:
    for _ in range(num_iterations):
        # 随机选择一个参数进行变异
        parameter_to_mutate = random.choice(list(current_parameters.keys()))
        # 如果参数是 'num_ants'，则直接设置为一个整数值
        if parameter_to_mutate == 'num_ants':
           new_value = random.randint(1, 100)  # 设置 'num_ants' 的随机整数值
        else:
           new_value = current_parameters[parameter_to_mutate] * random.uniform(0.8, 1.2)

        new_parameters = current_parameters.copy()
        new_parameters[parameter_to_mutate] = new_value

        # 计算成本差
        current_cost = evaluate_parameters(current_parameters)
        new_cost = evaluate_parameters(new_parameters)
        cost_difference = new_cost - current_cost

        # 判断是否接受变异
        if cost_difference < 0 or random.random() < math.exp(-cost_difference / temperature):
            current_parameters = new_parameters
            current_cost = new_cost

    # 降低温度
    temperature *= cooling_rate

# 打印最佳参数组合和成本
print("Best Parameters:", current_parameters)
print("Best Cost:", current_cost)
