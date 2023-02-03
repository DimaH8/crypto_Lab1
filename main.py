import os
import sys
import numpy

# if True, we will print tables on screen
verbose = True

# pretty-print
numpy.set_printoptions(suppress=True)
numpy.set_printoptions(formatter={'float': '{: 0.3f}'.format}, linewidth=150)

g_size = 20


def create_array():
    return numpy.zeros(g_size)


def create_empty_array():
    return numpy.empty(0)


def create_matrix(dt=float):
    return numpy.zeros((g_size, g_size), dtype=dt)


variants = [6, 10]
for variant in variants:
    print()
    print()
    print(f"variant = {variant}")
    probabilities = numpy.genfromtxt(f"prob_{variant}.csv", dtype=float, comments='#', delimiter=',')
    table = numpy.genfromtxt(f"table_{variant}.csv", dtype=int, comments='#', delimiter=',')

    # 1. Calculate P(C)
    P_C = create_array()
    for i in range(g_size):
        for j in range(g_size):
            index_c = table[i][j]
            # P(C_h) += P(k_i)*P(M_i)
            p_k_i = probabilities[0][j]
            p_M_i = probabilities[1][i]
            P_C[index_c] += p_k_i * p_M_i

    if verbose:
        print("P(C) =")
        print(P_C)

    # 2. Calculate Р(М, С)
    P_M_C = create_matrix()
    for i in range(g_size):
        for j in range(g_size):
            # P(C_table_xx[j][i], M_i) += P(M_i)*P(k_j)
            p_M_i = probabilities[0][i]
            p_k_j = probabilities[1][j]
            table_index = table[j][i]
            P_M_C[table_index][i] += p_M_i * p_k_j

    if verbose:
        print("Р(М, С) =")
        print(P_M_C)

    # 3. Calculate P(M | C)
    P_M_or_C = create_matrix()
    for x in range(g_size):
        P_M_or_C[x] = P_M_C[x] / P_C[x]

    if verbose:
        print("P(M | C) =")
        print(P_M_or_C)

    # 4. Calculate optimal stochastic solving function and optimal deterministic solving function
    opt_determistic = create_empty_array()
    opt_stochastic = create_matrix()
    for x in range(g_size):
        # Find indexes chiphertext where P(M|C) row has max values
        row_max_value = numpy.max(P_M_or_C[x])
        res = numpy.where(P_M_or_C[x] == row_max_value)
        index_max = res[0]
        # for this time (by our choice) we take only first index for optimal deterministic solving function
        opt_determistic = numpy.append(opt_determistic, index_max[0])
        if index_max.size == 1:
            opt_stochastic[x][index_max[0]] = 1
        else:
            for i in index_max:
                opt_stochastic[x][i] = 1 / index_max.size

    if verbose:
        print("Optimal stochastic solving function: ")
        print(opt_stochastic)
        print("Optimal deterministic solving function: ")
        print(opt_determistic)

    # 5. Calculating losses for deterministic solving function
    losses_data = create_matrix(int)
    for i in range(g_size):
        for j in range(g_size):
            index = table[i][j]
            if i != opt_determistic[index]:
                losses_data[index][i] = 1

    mean_losses_deterministic = numpy.sum(losses_data * P_M_C)
    print('Losses for deterministic solving function: mean value =', mean_losses_deterministic)

    if verbose:
        print("Losses for deterministic solving function: ")
        print(losses_data)

    # 6. Calculating losses for stochastic solving function
    losses_data = create_matrix()
    r = numpy.arange(g_size, dtype=int)
    for i in range(g_size):
        for j in range(g_size):
            losses_data[i][j] = numpy.sum(opt_stochastic[i][j + 1:]) + numpy.sum(opt_stochastic[i][:j])

    mean_losses_stochastic = numpy.sum(losses_data * P_M_C)
    print("Losses for stochastic solving function: mean value =", mean_losses_stochastic)

    if verbose:
        print("Losses for stochastic solving function: ")
        print(losses_data)

