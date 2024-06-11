import numpy as np
import random

# 假设航班时间窗是在每个机场的每天早上6点到晚上10点之间
FLIGHT_WINDOW_START = 6
FLIGHT_WINDOW_END = 22

# 定义飞机资源约束
class Aircraft:
    def __init__(self, available_time, maintenance_interval):
        self.available_time = available_time
        self.maintenance_interval = maintenance_interval

# 更新蚁群算法
class AntColony:
    def __init__(self, flights, delay_data, aircraft_data, num_ants=10, max_iter=100, evaporation_rate=0.5, alpha=1, beta=2):
        self.flights = flights
        self.delay_data = delay_data
        self.aircraft_data = aircraft_data
        self.num_ants = num_ants
        self.max_iter = max_iter
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.num_flights = len(flights)
        self.pheromone = np.ones(self.num_flights)  # 初始化信息素浓度

    # 更新信息素
    def update_pheromone(self, path, total_delay):
        self.pheromone *= (1 - self.evaporation_rate)  # 信息素挥发
        for flight in path:
            self.pheromone[flight] += 1 / (1 + total_delay)  # 根据延误时间更新信息素

    # 选择下一个航班
    def select_next_flight(self, available_flights, current_time, aircraft):
        probabilities = (self.pheromone ** self.alpha) * ((1 / (1 + self.delay_data.iloc[available_flights]["Delay"].values)) ** self.beta)
        probabilities *= self.get_heuristic_probabilities(available_flights, current_time, aircraft)
        probabilities /= probabilities.sum()  # 归一化概率
        return np.random.choice(available_flights, p=probabilities)

    # 获取启发式概率
    def get_heuristic_probabilities(self, available_flights, current_time, aircraft):
        heuristic_probabilities = []
        for flight_index in available_flights:
            flight = self.flights.iloc[flight_index]
            if current_time + flight["Distance"] / 500 <= FLIGHT_WINDOW_END and current_time + flight["Distance"] / 500 <= aircraft.available_time:
                heuristic_probabilities.append(1)
            else:
                heuristic_probabilities.append(0)
        return np.array(heuristic_probabilities)

    # 运行蚁群算法
    def run(self):
        best_path = None
        best_delay = float('inf')

        for _ in range(self.max_iter):
            total_delay = 0
            paths = []
            for ant in range(self.num_ants):
                path = []
                current_time = FLIGHT_WINDOW_START
                current_aircraft = random.choice(self.aircraft_data)
                available_flights = list(range(self.num_flights))
                while available_flights:
                    next_flight = self.select_next_flight(available_flights, current_time, current_aircraft)
                    available_flights.remove(next_flight)
                    current_time += self.flights.iloc[next_flight]["Distance"] / 500
                    path.append(next_flight)
                delay_sum = self.delay_data.iloc[path]["Delay"].sum()  # 修改这里
                total_delay += delay_sum
                paths.append((path, delay_sum))
            if total_delay < best_delay:
                best_delay = total_delay
                best_path = paths[np.argmin([p[1] for p in paths])][0]
            self.update_pheromone(best_path, best_delay)
        
        return best_path, best_delay



# 使用蚁群算法优化航班排序
def optimize_flight_schedule(flight_data, delay_data, aircraft_data):
    ant_colony = AntColony(flight_data, delay_data, aircraft_data)
    best_path, best_delay = ant_colony.run()
    return best_path, best_delay

# 测试优化函数
num_flights = 1500
num_airports = 20
num_aircraft = 50
best_path, best_delay = optimize_flight_schedule(flight_data, delay_data, aircraft_data)
print("最佳航班顺序:", best_path)
print("最佳延误时间:", best_delay)
