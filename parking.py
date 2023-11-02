import math
import random
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
K = 200
mutation_rate = 0.005


def to_range(val, i):
    """
    To bring the decimal values in the range of expected Gamma and Beta Values
    :param val: given decimal value
    :param i: Index to determine whether it's a gama or a beta value
    :return: The converted gamma/beta value
    """
    if i%2 == 0:
        return (val*1.048)/127 + (-0.524)
    else:
        return (val*10)/127 + (-5)


def interpolate(converted_list):
    """
    Function to implement Cubic Spline
    :param converted_list: List with beta/gamma values
    :return: Interpolated List
    """
    y_gamma = list()
    y_beta = list()
    x = np.linspace(0, 10, 10)
    interpolation_val = 100
    x_intr = np.linspace(0, 10, interpolation_val)
    for i in range(len(converted_list)):
        if i%2 == 0:
            y_gamma.append(converted_list[i])
        else:
            y_beta.append(converted_list[i])
    cs_gamma = CubicSpline(x, y_gamma)
    cs_beta = CubicSpline(x, y_beta)
    intr_gamma = cs_gamma(x_intr)
    intr_beta = cs_beta(x_intr)
    intr_list = list()
    for i in range(interpolation_val):
        intr_list.append(intr_gamma[i])
        intr_list.append(intr_beta[i])
    return intr_list


def feasible(x, y):
    """
    Check the feasibility range for given state vector
    :param x: Position of x
    :param y: Position of y
    :return: Boolean value toh determine feasibility
    """
    if x <= -4 and y > 3:
        return True
    elif (-4 < x < 4) and y > -1:
        return True
    elif x >= 4 and y > 3:
        return True
    else:
        return False


def euclidean_dist(start_pos, end_pos):
    """
    Calculate the Euclidean Distance
    :param start_pos: Derived end position
    :param end_pos: Expected end position
    :return: the Euclidean distance
    """
    return math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2 + (end_pos[2] - start_pos[2])**2 + (end_pos[3] - start_pos[3])**2)


def cost_function(interpolated_list):
    """
    Cost Function for a given population
    :param interpolated_list: The interpolated list
    :return: The Euclidean Distance
    """
    flag = 0
    h = 0.1

    start_pos = [0, 8, 0, 0]
    end_pos = [0, 0, 0, 0]
    for i in range(0, len(interpolated_list), 2):
        x_i = start_pos[0] + h*(start_pos[3]*math.cos(start_pos[2]))
        y_i = start_pos[1] + h*(start_pos[3]*math.sin(start_pos[2]))
        aplha_i = start_pos[2] + h*interpolated_list[i]
        velocity_i = start_pos[3] + h*interpolated_list[i+1]
        start_pos = [x_i, y_i, aplha_i, velocity_i]
        if not feasible(start_pos[0], start_pos[1]):
            flag = 1
            break
    if flag == 1:
        return K
    else:
        return euclidean_dist(start_pos, end_pos)


def elitism(parent_list, fitness_ratio):
    """
    Perform Elitism i.e extract 2 parents with the highest fitness ratio
    :param parent_list: The population list
    :param fitness_ratio: fitnedd ratio list
    :return:
    """
    parent1 = parent_list[fitness_ratio.index(max(fitness_ratio))]
    parent_list.pop(fitness_ratio.index(max(fitness_ratio)))
    parent2 = parent_list[fitness_ratio.index(max(fitness_ratio))]
    parent_list.pop(fitness_ratio.index(max(fitness_ratio)))
    return parent1, parent2


def list_to_string(list1):
    """
    Convert a list to string
    :param list1: given list
    :return: String
    """
    str1 = ""
    return str1.join(list1)


def crossover(parent_list, fitness_ratio):
    """
    Perform Crossover and Mutation and generate the next generation
    :param parent_list: Current generation list
    :param fitness_ratio: list of fitness ratio
    :return: new generation
    """
    crossover_point = random.randint(0, 140)
    new_parent = list()
    for i in range(99):
        parent1, parent2 = random.choices(parent_list, fitness_ratio, k = 2)
        ele1 = list_to_string(parent1)
        ele2 = list_to_string(parent2)
        child1 = ele1[:crossover_point] + ele2[crossover_point:]
        child2 = ele2[:crossover_point] + ele1[crossover_point:]
        temp1 = ""
        temp2 = ""
        temp = 0
        child1_list = list()
        child2_list = list()
        for j in range(len(child1)):
            val = random.uniform(0,1)
            if val < mutation_rate:
                temp1 += str((int(child1[j])+1)%2)
                temp2 += str((int(child2[j]) + 1) % 2)
            else:
                temp1 += child1[j]
                temp2 += child2[j]

            if (j+1)%7 == 0:
                child1_list.append(temp1[temp:j+1])
                child2_list.append(temp2[temp:j+1])
                temp = j + 1
        new_parent.append(child1_list)
        new_parent.append(child2_list)
    elite1, elite2 = elitism(parent_list, fitness_ratio)
    new_parent.append(elite1)
    new_parent.append(elite2)

    return new_parent


def binaryToDecimal(n):
    """
    Convert binary to decimal value
    :param n: binary value
    :return: its decimal equivalent
    """
    return int(n, 2)


def initial_population(population_size):
    """
    Creating the first generation randomly and generating next generation
    :return:
    """
    J_list = list()
    fitness_function = list()
    fitness_ratio = list()
    parent_list = list()
    for i in range(population_size):
        decimal_list = list()
        for j in range(0, 20):
            decimal_list.append(random.randint(0, 127))
        converted_list = list()
        binary_list = list()
        for j in range(0, 20):
            binary_list.append(format(decimal_list[j], '07b'))
            converted_list.append(to_range(decimal_list[j], j))
        parent_list.append(binary_list)
        interpolated_list = interpolate(converted_list)
        J_list.append(cost_function(interpolated_list))
        fitness_function.append(1 / (J_list[i] + 1))
    sum_ = sum(fitness_function)
    for fn in range(len(fitness_function)):
        fitness_ratio.append((fitness_function[fn] / sum_) * 100)
    print("Generation " + str(0) + " : J = " + str(min(J_list)))
    return parent_list, fitness_ratio


def generation(parent_list, fitness_ratio, population_size, tolerance):
    """
    Running all the generations
    :param tolerance: tolerance threshold
    :param population_size: population size
    :param parent_list: Current generation list
    :param fitness_ratio: list of fitness ratio
    :return:
    """
    for i in range(1, 1200):
        parent_list = crossover(parent_list, fitness_ratio)
        fitness_function = list()
        fitness_ratio = list()
        J_list = list()
        for k in range(population_size):
            decimal_list = list()
            for j in range(0, 20):
                decimal_list.append(binaryToDecimal(parent_list[k][j]))
            converted_list = list()
            for j in range(0, 20):
                converted_list.append(to_range(decimal_list[j], j))
            interpolated_list = interpolate(converted_list)
            J_list.append(cost_function(interpolated_list))
            fitness_function.append(1 / (J_list[k] + 1))
        sum_ = sum(fitness_function)
        for fn in range(len(fitness_function)):
            fitness_ratio.append((fitness_function[fn] / sum_)*100)
        print("Generation " + str(i) + " : J = " + str((min(J_list))))
        if min(J_list) < tolerance:
            break
    return interpolated_list, converted_list


def plot_trajectory(state_vector):
    """
    Plot for trajectory
    :param state_vector: State vector for best generation
    :return:
    """
    x_val = list()
    y_val = list()
    for i in range(len(state_vector)):
        x_val.append(state_vector[i][0])
        y_val.append(state_vector[i][1])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim([-15, 15])
    ax.set_ylim([-10, 15])

    plt.plot([-15, -4], [3, 3], color='black')
    plt.plot([-4, -4], [3, -1], color='black')
    plt.plot([-4, 4], [-1, -1], color='black')
    plt.plot([4, 4], [-1, 3], color='black')
    plt.plot([4, 15], [3, 3], color='black')

    plt.plot(x_val, y_val, color='green')
    plt.title('State trajectory with obstacles')
    ax.set_xlabel('x (ft)')
    ax.set_ylabel('y (ft)')
    ax.grid(True)

    plt.show()


def plot_state_vectors(state_vector):
    """
    Plot for state vector
    :param state_vector: State vector for best generation
    :return:
    """
    time = np.linspace(0,10,101)
    x_val = list()
    y_val = list()
    alpha_val = list()
    v_val = list()
    for i in range(len(state_vector)):
        x_val.append(state_vector[i][0])
        y_val.append(state_vector[i][1])
        alpha_val.append(state_vector[i][2])
        v_val.append(state_vector[i][3])

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(time, x_val)
    axs[0, 0].set_title('State Vector - x')
    axs[0, 0].set(ylabel='x (ft)', xlabel='Time (s)')
    axs[0, 1].plot(time, y_val, 'tab:orange')
    axs[0, 1].set_title('State Vector - y')
    axs[0, 1].set(ylabel='y (ft)', xlabel='Time (s)')
    axs[1, 0].plot(time, alpha_val, 'tab:green')
    axs[1, 0].set_title('State Vector - alpha')
    axs[1, 0].set(ylabel='alpha (rad)', xlabel='Time (s)')
    axs[1, 1].plot(time, v_val, 'tab:red')
    axs[1, 1].set_title('State Vector - v')
    axs[1, 1].set(ylabel='v (ft/s)', xlabel='Time (s)')

    for ax in axs.flat:
        ax.label_outer()
    plt.show()


def plot_control_histories(control_vector):
    """
    Plot for control vector
    :param control_vector: control vector for best generation
    :return:
    """
    time = np.linspace(0,10,101)
    gamma_val = [0]
    beta_val = [0]
    for i in range(0, len(control_vector), 2):
        gamma_val.append(control_vector[i])
        beta_val.append(control_vector[i+1])
    fig, axs = plt.subplots(2)
    fig.suptitle('Control Histories')
    axs[0].plot(time, gamma_val)
    axs[0].set(ylabel='gamma (rad/s)', xlabel='Time (s)')
    axs[1].plot(time, beta_val)
    axs[1].set(ylabel='beta (ft/s^2)', xlabel='Time (s)')
    plt.show()


def generate_state_vector(control_vector, population_size):
    """
    Generate State vectors for a given trajectory
    :param control_vector: control vector list
    :return:
    """
    h = 0.1
    state_vector = list()
    start_pos = [0, 8, 0, 0]
    state_vector.append(start_pos)
    for i in range(0, population_size, 2):
        x_i = start_pos[0] + h * (start_pos[3] * math.cos(start_pos[2]))
        y_i = start_pos[1] + h * (start_pos[3] * math.sin(start_pos[2]))
        alpha_i = start_pos[2] + h * control_vector[i]
        velocity_i = start_pos[3] + h * control_vector[i + 1]
        start_pos = [x_i, y_i, alpha_i, velocity_i]
        state_vector.append(start_pos)
    return state_vector


def main():
    """
    Driver Function
    :return:
    """
    population_size = 200
    tolerance = 0.1
    parent_list, fitness_ratio = initial_population(population_size)
    control_vector, control_vals = generation(parent_list, fitness_ratio, population_size, tolerance)
    state_vector = generate_state_vector(control_vector, population_size)
    print()
    print("Final State Values:")
    print("x_f = " + str(state_vector[-1][0]))
    print("y_f = " + str(state_vector[-1][1]))
    print("alpha_f = " + str(state_vector[-1][2]))
    print("v_f = " + str(state_vector[-1][3]))
    plot_trajectory(state_vector)
    plot_state_vectors(state_vector)
    plot_control_histories(control_vector)
    file1 = open("control.dat", "w")
    for i in control_vals:
        file1.write(str(i)+"\n")
    file1.close()


if __name__ == "__main__":
    main()