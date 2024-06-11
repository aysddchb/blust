# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 20:56:22 2024

@author: 86182
"""

import random
import pandas as pd
from datetime import datetime, timedelta

# 全局变量
airport_distances = {}  # 保存机场对之间的距离

# 生成航班数据集
def generate_flights(num_flights, num_airports):
    flights = []
    for i in range(num_flights):
        departure_airport = f'AIR{random.randint(0, num_airports-1)}'
        arrival_airport = f'AIR{random.randint(0, num_airports-1)}'
        
        # 确保相同机场对之间的距离一致
        if (departure_airport, arrival_airport) not in airport_distances:
            airport_distances[(departure_airport, arrival_airport)] = random.randint(100, 2500)
        
        flight_id = f'Flight{i+1}'
        departure_time = random_datetime()
        arrival_time = departure_time + timedelta(minutes=random.randint(60, 600))
        wind_speed = random.randint(0, 50)
        weather = random.choice(["sunny", "rainy", "stormy"])
        
        # 根据风速和天气调整实际出发时间和实际到达时间
        actual_departure_time = adjust_time(departure_time, wind_speed, weather)
        actual_arrival_time = adjust_time(arrival_time, wind_speed, weather)
        
        # 判断是否延误
        delayed = actual_departure_time > departure_time or actual_arrival_time > arrival_time
        
        # 计算损失成本
        loss_cost = calculate_loss_cost(delayed, departure_time, arrival_time, actual_departure_time, actual_arrival_time)
        
        aircraft_type = random.choice(["A319", "Boeing 737", "A340", "A321", "Boeing 777", "A320", "A380"])
        
        flights.append({
            'Flight_ID': flight_id,
            'Departure_Time': departure_time,
            'Arrival_Time': arrival_time,
            'Actual_Departure_Time': actual_departure_time,
            'Actual_Arrival_Time': actual_arrival_time,
            'Departure_Airport': departure_airport,
            'Arrival_Airport': arrival_airport,
            'Flight_Distance': airport_distances[(departure_airport, arrival_airport)],
            'Aircraft_Type': aircraft_type,
            'Delayed': delayed,
            'Wind_Speed': wind_speed,
            'Weather': weather,
            'Loss_Cost': loss_cost
        })
    
    return pd.DataFrame(flights)

# 生成随机日期时间
def random_datetime():
    start = datetime(2023, 1, 1)
    end = datetime(2023, 12, 31)
    return start + timedelta(minutes=random.randint(0, int((end - start).total_seconds() / 60)))

# 根据风速和天气调整时间
def adjust_time(base_time, wind_speed, weather):
    # 当风速越大、天气越极端时，时间越比预计晚
    if weather == "stormy":
        delay_factor = 1.5
    elif weather == "rainy":
        delay_factor = 1.2
    else:
        delay_factor = 1.0
    
    delay_minutes = wind_speed * 0.1 * delay_factor
    adjusted_time = base_time + timedelta(minutes=delay_minutes)
    return adjusted_time

# 计算损失成本
def calculate_loss_cost(delayed, departure_time, arrival_time, actual_departure_time, actual_arrival_time):
    # 不延误的情况下没有损失成本
    if not delayed:
        return 0
    
    # 延误时间越长，损失成本数目概率越大
    delay_minutes = (actual_departure_time - departure_time).total_seconds() / 60
    delay_minutes += (actual_arrival_time - arrival_time).total_seconds() / 60
    loss_cost = delay_minutes * random.randint(100, 200)
    return loss_cost

# 生成机场数据集
def generate_airports(num_airports):
    airport_data = {
        'Airport_Code': [f'AIR{i}' for i in range(num_airports)],
        'Capacity': [random.randint(50, 200) for _ in range(num_airports)],
        'Flight_Traffic': [random.randint(10, 100) for _ in range(num_airports)],
        'Runway_Utilization': [random.uniform(0.5, 0.9) for _ in range(num_airports)]
    }
    return pd.DataFrame(airport_data)

# 生成飞机数据集
def generate_aircrafts(num_aircrafts):
    aircraft_data = {
        'Aircraft_ID': [f'Aircraft{i+1}' for i in range(num_aircrafts)],
        'Available_Time': [datetime.strptime("2023-01-01 00:00:00", "%Y-%m-%d %H:%M:%S") + timedelta(minutes=random.randint(0, 1440)) for _ in range(num_aircrafts)],
        'Maintenance_Requirement': [random.randint(1, 5) for _ in range(num_aircrafts)],
        'Passenger_Capacity': [random.randint(100, 300) for _ in range(num_aircrafts)]
    }
    return pd.DataFrame(aircraft_data)

# 生成数据集
def generate_dataset(num_flights, num_airports, num_aircrafts):
    flights = generate_flights(num_flights, num_airports)
    airports = generate_airports(num_airports)
    aircrafts = generate_aircrafts(num_aircrafts)
    return flights, airports, aircrafts

# 初始化数据集
num_flights = 1500
num_airports = 10
num_aircrafts = 20
flights, airports, aircrafts = generate_dataset(num_flights, num_airports, num_aircrafts)

# 打印数据集示例
print("Flights:")
print(flights.head())

print("\nAirports:")
print(airports.head())

print("\nAircrafts:")
print(aircrafts.head())
flights.to_csv(r'D:\python\兼职\5.31/' +"flights.csv", index=False)
airports.to_csv(r'D:\python\兼职\5.31/' +"airports.csv", index=False)
aircrafts.to_csv(r'D:\python\兼职\5.31/' +"aircrafts.csv", index=False)


# 更新最佳航班顺序和最佳延误时间
    if total_delay < best_delay_time:
        best_flight_order = visited_flights
        best_delay_time = total_delay

# 更新信息素
for i in range(num_flights):
    for j in range(num_flights):
        if j in best_flight_order:  # 如果下一个航班在最佳顺序中
            pheromones[i, j] *= (1 - evaporation_rate)  # 蒸发
            pheromones[i, j] += alpha / (1 + best_delay_time)  # 信息素增加