import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_delay(flights):
    # 假设flights包含'Departure_Time'、'Actual_Departure_Time'、'Arrival_Time'和'Actual_Arrival_Time'列
    flights['Departure_Time'] = pd.to_datetime(flights['Departure_Time'])
    flights['Actual_Departure_Time'] = pd.to_datetime(flights['Actual_Departure_Time'])
    flights['Arrival_Time'] = pd.to_datetime(flights['Arrival_Time'])
    flights['Actual_Arrival_Time'] = pd.to_datetime(flights['Actual_Arrival_Time'])

    # 计算出发延误时间（单位：分钟）
    flights['Departure_Delay'] = (flights['Actual_Departure_Time'] - flights['Departure_Time']).dt.total_seconds() / 60

    # 计算到达延误时间（单位：分钟）
    flights['Delay'] = (flights['Actual_Arrival_Time'] - flights['Arrival_Time']).dt.total_seconds() / 60

    return flights

def ant_colony_optimization(flights, num_iterations=7, num_ants=100, evaporation_rate=0.9, alpha=2.0, beta=1.5):
    num_flights = len(flights)
    pheromones = np.ones((num_flights, num_flights))
    total_delay_history = []
    best_flight_order = []
    best_delay_time = float('inf')
    convergence_curve = []

    for iteration in range(num_iterations):
        # 每轮迭代放置蚂蚁
        for ant in range(num_ants):
            # 每只蚂蚁从一个航班开始
            current_flight = random.randint(0, num_flights - 1)
            visited_flights = [current_flight]  # 记录已访问的航班
            total_delay = 0  # 记录当前蚂蚁选择的航班顺序的总延误

            # 选择下一个航班，直到所有航班都被访问过
            while len(visited_flights) < num_flights:
                # 计算当前航班的延误
                current_delay = flights.at[current_flight, 'Delay']
                total_delay += current_delay * random.uniform(0,1.2)

                # 计算下一个航班的可能性
                probabilities = []
                for i in range(num_flights):
                    if i not in visited_flights:
                        pheromone = pheromones[current_flight, i]
                        delay = flights.at[i, 'Delay']
                        probability = pheromone ** alpha / (1 + delay) ** beta
                        probabilities.append((i, probability))

                # 根据概率选择下一个航班
                total_probability = sum(probability for _, probability in probabilities)
                threshold = random.uniform(0, total_probability)
                cumulative_probability = 0
                for next_flight, probability in probabilities:
                    cumulative_probability += probability
                    if cumulative_probability >= threshold:
                        current_flight = next_flight
                        visited_flights.append(next_flight)
                        break

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

        # 记录总延误信息
        total_delay_history.append(best_delay_time)
        convergence_curve.append(iteration)

        # 打印关键信息
        print(f"Iteration: {iteration + 1}/{num_iterations}, Best delay: {best_delay_time}")

    return total_delay_history, best_flight_order, best_delay_time, convergence_curve, pheromones

# 读取航班数据
flights = pd.read_csv(r'D:\python\兼职\5.31/' +"flights.csv")
flights = calculate_delay(flights)

# 蚁群算法优化
total_delay_history, best_flight_order, best_delay_time, convergence_curve, pheromones = ant_colony_optimization(flights)

# 计算降低的成本
initial_total_delay = flights['Delay'].sum()
final_total_delay = total_delay_history[-1]
reduced_cost = initial_total_delay - final_total_delay

print(f"Final total delay: {final_total_delay}")
print(f"Best flight order: {best_flight_order}")
print(f"Best delay time: {best_delay_time}")
print(f"Reduced cost: {reduced_cost}")

# 收敛曲线图
plt.plot(convergence_curve, total_delay_history)
plt.xlabel('Iteration')
plt.ylabel('Best Delay')
plt.title('Convergence Curve')
plt.show()

# 信息素分布图
# plt.figure(figsize=(10, 8))
# sns.heatmap(pheromones, cmap='YlGnBu', annot=True, fmt=".2f")
# plt.xlabel('Next Flight')
# plt.ylabel('Current Flight')
# plt.title('Pheromone Distribution')
# plt.show()

# 敏感性分析函数
def sensitivity_analysis(flights, param_name, param_values, **kwargs):
    results = []
    for value in param_values:
        kwargs[param_name] = value
        _, _, best_delay_time, _, _ = ant_colony_optimization(flights, **kwargs)
        results.append(best_delay_time)
    return results

# 参数调优曲线图
def plot_parameter_tuning_curve(param_name, param_values, results):
    plt.plot(param_values, results, marker='o')
    plt.xlabel(param_name)
    plt.ylabel('Best Delay')
    plt.title(f'Parameter Tuning: {param_name}')
    plt.show()

# 蚂蚁路径可视化
def visualize_ant_paths(flights, best_flight_order):
    plt.figure(figsize=(10, 8))
    for i in range(len(best_flight_order) - 1):
        start_flight = best_flight_order[i]
        end_flight = best_flight_order[i + 1]
        plt.plot([flights.at[start_flight, 'Departure_Time'], flights.at[end_flight, 'Arrival_Time']],
                 [i, i], marker='o')
    plt.xlabel('Time')
    plt.yticks(range(len(best_flight_order)), [f'Flight {flight}' for flight in best_flight_order])
    plt.title('Ant Paths')
    plt.show()

# 定义参数范围
num_iterations_values = [5, 10, 15, 20, 25]
num_ants_values = [50, 100, 150, 200, 250]
evaporation_rate_values = [0.7, 0.8, 0.9, 0.95, 0.99]
alpha_values = [1.0, 1.5, 2.0, 2.5, 3.0]
beta_values = [1.0, 1.5, 2.0, 2.5, 3.0]

# 敏感性分析：迭代次数
iteration_results = sensitivity_analysis(flights, 'num_iterations', num_iterations_values)
plot_parameter_tuning_curve('Number of Iterations', num_iterations_values, iteration_results)

# 敏感性分析：蚂蚁数量
ants_results = sensitivity_analysis(flights, 'num_ants', num_ants_values)
plot_parameter_tuning_curve('Number of Ants', num_ants_values, ants_results)

# 敏感性分析：信息素挥发率
evaporation_results = sensitivity_analysis(flights, 'evaporation_rate', evaporation_rate_values)
plot_parameter_tuning_curve('Evaporation Rate', evaporation_rate_values, evaporation_results)

# 敏感性分析：启发式因子 alpha
alpha_results = sensitivity_analysis(flights, 'alpha', alpha_values)
plot_parameter_tuning_curve('Alpha', alpha_values, alpha_results)

# 敏感性分析：启发式因子 beta
beta_results = sensitivity_analysis(flights, 'beta', beta_values)
plot_parameter_tuning_curve('Beta', beta_values, beta_results)

# 蚂蚁路径可视化
_, best_flight_order, _, _ = ant_colony_optimization(flights)
visualize_ant_paths(flights, best_flight_order)
