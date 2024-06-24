# import numpy as np
# import scipy.stats as stats
# import scipy.integrate as integrate
# import itertools
# import multiprocessing as mp
# from collections import defaultdict

# # Ensure correct number of threads
# num_threads = mp.cpu_count()
# print(f"Number of threads: {num_threads}")

# # Define distributions using parameters
# mu_00 = mu_11 = 1
# mu_01 = mu_10 = -1
# sigma_00 = sigma_01 = sigma_10 = sigma_11 = 1

# # Statistical functions
# f_opt = lambda x: stats.norm.pdf(x, mu_00, sigma_00)
# F_opt = lambda x: stats.norm.cdf(x, mu_00, sigma_00)
# f_sub = lambda x: stats.norm.pdf(x, mu_01, sigma_01)
# F_sub = lambda x: stats.norm.cdf(x, mu_01, sigma_01)
# g_opt = lambda v: stats.norm.pdf(v, mu_11, sigma_11)
# G_opt = lambda v: stats.norm.cdf(v, mu_11, sigma_11)
# g_sub = lambda v: stats.norm.pdf(v, mu_10, sigma_10)
# G_sub = lambda v: stats.norm.cdf(v, mu_10, sigma_10)

# # Precompute integrals for h functions if they are often reused
# h_cache = defaultdict(float)

# def h_00(K, i):
#     key = ("h_00", K, i)
#     if key in h_cache:
#         return h_cache[key]
    
#     def f(x, v):
#         return f_opt(x) * (K - i) * F_opt(x)**(K - i - 1) * g_sub(v) * i * G_sub(v)**(i - 1)
    
#     result, _ = integrate.dblquad(f, -np.inf, np.inf, lambda v: -np.inf, lambda v: v)
#     h_cache[key] = 1 - result
#     return h_cache[key]

# def h_01(K, i):
#     key = ("h_01", K, i)
#     if key in h_cache:
#         return h_cache[key]
    
#     def f(x, v):
#         return f_sub(x) * (K - i) * F_sub(x)**(K - i - 1) * g_opt(v) * i * G_opt(v)**(i - 1)
    
#     result, _ = integrate.dblquad(f, -np.inf, np.inf, lambda v: -np.inf, lambda v: v)
#     h_cache[key] = 1 - result
#     return h_cache[key]

# def h_10(K, i):
#     key = ("h_10", K, i)
#     if key in h_cache:
#         return h_cache[key]
    
#     def f(v, x):
#         return g_sub(v) * i * G_sub(v)**(i - 1) * f_opt(x) * (K - i) * F_opt(x)**(K - i - 1)
    
#     result, _ = integrate.dblquad(f, -np.inf, np.inf, lambda x: -np.inf, lambda x: x)
#     h_cache[key] = 1 - result
#     return h_cache[key]

# def h_11(K, i):
#     key = ("h_11", K, i)
#     if key in h_cache:
#         return h_cache[key]
    
#     def f(v, x):
#         return g_opt(v) * i * G_opt(v)**(i - 1) * f_sub(x) * (K - i) * F_sub(x)**(K - i - 1)
    
#     result, _ = integrate.dblquad(f, -np.inf, np.inf, lambda x: -np.inf, lambda x: x)
#     h_cache[key] = 1 - result
#     return h_cache[key]

# def N(i, q_hat, gamma_hat, phi_hat, A, J, K, mean_00, mean_10):
#     n1 = phi_hat * (1 - gamma_hat) * (1 - q_hat)**i * q_hat**(K - i)
#     n2 = (1 - phi_hat) * gamma_hat * q_hat**i * (1 - q_hat)**(K - i)
#     n3 = (1 - phi_hat) * (1 - gamma_hat) * (1 - q_hat)**i * q_hat**(K - i)
#     n4 = phi_hat * gamma_hat * q_hat**i * (1 - q_hat)**(K - i)
#     result = 0.0

#     if A == 1 and J == 1:
#         if i == 0:
#             result = 0.0
#         elif 0 < i < K:
#             result = n1 * h_10(K, i) + n2 * h_11(K, i)
#         else:  # i == K
#             result = n1 + n2
#     elif A == 1 and J == 0:
#         if i == 0:
#             result = n1 + n2
#         elif 0 < i < K:
#             result = n1 * h_00(K, i) + n2 * h_01(K, i)
#         else:  # i == K
#             result = 0.0
#     elif A == 0:
#         if J == 1:
#             if i == 0:
#                 result = 0.0
#             elif 0 < i < K:
#                 result = n3 * h_10(K, i) + n4 * h_11(K, i)
#             else:  # i == K
#                 result = n3 + n4
#         else:  # J == 0
#             if i == 0:
#                 result = n3 + n4
#             elif 0 < i < K:
#                 result = n3 * h_00(K, i) + n4 * h_01(K, i)
#             else:  # i == K
#                 result = 0.0
    
#     return result * (mean_00 - mean_10)

# def B(i, q_hat, gamma_hat, phi_hat, A, J, K, mean_01, mean_11):
#     b1 = (1 - phi_hat) * gamma_hat * (1 - q_hat)**i * q_hat**(K - i)
#     b2 = phi_hat * (1 - gamma_hat) * q_hat**i * (1 - q_hat)**(K - i)
#     b3 = phi_hat * gamma_hat * (1 - q_hat)**i * q_hat**(K - i)
#     b4 = (1 - phi_hat) * (1 - gamma_hat) * q_hat**i * (1 - q_hat)**(K - i)
#     result = 0.0

#     if A == 1 and J == 1:
#         if i == 0:
#             result = 0.0
#         elif 0 < i < K:
#             result = b1 * h_10(K, i) + b2 * h_11(K, i)
#         else:  # i == K
#             result = b1 + b2
#     elif A == 1 and J == 0:
#         if i == 0:
#             result = b1 + b2
#         elif 0 < i < K:
#             result = b1 * h_00(K, i) + b2 * h_01(K, i)
#         else:  # i == K
#             result = 0.0
#     elif A == 0:
#         if J == 1:
#             if i == 0:
#                 result = 0.0
#             elif 0 < i < K:
#                 result = b3 * h_10(K, i) + b4 * h_11(K, i)
#             else:  # i == K
#                 result = b3 + b4
#         else:  # J == 0
#             if i == 0:
#                 result = b3 + b4
#             elif 0 < i < K:
#                 result = b3 * h_00(K, i) + b4 * h_01(K, i)
#             else:  # i == K
#                 result = 0.0
    
#     return result * (mean_11 - mean_01)

# def signal_ratio(s, rho_hat):
#     return rho_hat / (1 - rho_hat) if s == 1 else (1 - rho_hat) / rho_hat

# def choice_1(signal_ratio, N, B):
#     return 0 if B == 0 else (1 if signal_ratio > N / B else 0)

# def compute_choices(g_idx, genotype_matrix, Omega_matrix):
#     rho_hat, q_hat, gamma_hat, phi_hat = genotype_matrix[:, g_idx]
#     choice_matrix_row = np.zeros(Omega_matrix.shape[1])
#     for o_idx in range(Omega_matrix.shape[1]):
#         i, s, a, j = Omega_matrix[:, o_idx]
#         N_value = N(i, q_hat, gamma_hat, phi_hat, a, j, 3, mu_00, mu_10)
#         B_value = B(i, q_hat, gamma_hat, phi_hat, a, j, 3, mu_01, mu_11)
#         signal_ratio_value = signal_ratio(s, rho_hat)
#         C_value = choice_1(signal_ratio_value, N_value, B_value)
#         choice_matrix_row[o_idx] = C_value
#     return choice_matrix_row

# def main_computation():
#     K = 3
#     i_values = range(K + 1)
#     s_values = [0, 1]
#     a_values = [0, 1]
#     j_values = [0, 1]

#     combinations = []
#     for i in i_values:
#         if i == 0:
#             combinations.extend([[i, s, a, 0] for s, a in itertools.product(s_values, a_values)])
#         elif i == K:
#             combinations.extend([[i, s, a, 1] for s, a in itertools.product(s_values, a_values)])
#         else:
#             combinations.extend([[i, s, a, j] for j, s, a in itertools.product(j_values, s_values, a_values)])

#     Omega_matrix = np.array(combinations).T
#     print(f"Shape of the Omega matrix: {Omega_matrix.shape}")

#     rho_hat_values = np.linspace(0.5001, 0.9999, 6)
#     q_hat_values = np.linspace(0.0001, 0.9999, 11)
#     gamma_hat_values = np.linspace(0.0001, 0.9999, 11)
#     phi_hat_values = np.linspace(0.0001, 0.9999, 11)

#     combinations_gene = list(itertools.product(rho_hat_values, q_hat_values, gamma_hat_values, phi_hat_values))
#     genotype_matrix = np.array([list(comb) for comb in combinations_gene]).T
#     print(f"Shape of the genotype matrix: {genotype_matrix.shape}")

#     with mp.Pool(num_threads) as pool:
#         args = [(g_idx, genotype_matrix, Omega_matrix) for g_idx in range(genotype_matrix.shape[1])]
#         choice_matrix = np.array(pool.starmap(compute_choices, args))

#     return choice_matrix

# if __name__ == "__main__":
#     import time
#     start_time = time.time()
#     choice_matrix = main_computation()
#     print(f"First 5 rows of the choice matrix:\n{choice_matrix[:5, :]}")

#     # Save and load functionality
#     def save_data(data, filename):
#         np.save(filename, data)

#     def load_data(filename):
#         return np.load(filename)

#     # Example usage
#     choice_file = "choice_matrix.npy"
#     save_data(choice_matrix, choice_file)
#     loaded_matrix = load_data(choice_file)

#     if loaded_matrix.size > 0:
#         print("Choice matrix loaded successfully.")
#         print(f"Shape of the matrix: {loaded_matrix.shape}")
#     else:
#         print("Choice matrix empty.")

#     reduced_matrix = np.unique(loaded_matrix, axis=0)
#     print(f"Shape of the reduced result matrix: {reduced_matrix.shape}")

#     print(f"Execution time: {time.time() - start_time} seconds")




import numpy as np
import scipy.stats as stats
import scipy.integrate as integrate
import itertools
import multiprocessing as mp
from collections import defaultdict

# Ensure correct number of threads
num_threads = mp.cpu_count()
print(f"Number of threads: {num_threads}")

# Define distributions using parameters
mu_00 = mu_11 = 1
mu_01 = mu_10 = -1
sigma_00 = sigma_01 = sigma_10 = sigma_11 = 1

# Statistical functions
f_opt = lambda x: stats.norm.pdf(x, mu_00, sigma_00)
F_opt = lambda x: stats.norm.cdf(x, mu_00, sigma_00)
f_sub = lambda x: stats.norm.pdf(x, mu_01, sigma_01)
F_sub = lambda x: stats.norm.cdf(x, mu_01, sigma_01)
g_opt = lambda v: stats.norm.pdf(v, mu_11, sigma_11)
G_opt = lambda v: stats.norm.cdf(v, mu_11, sigma_11)
g_sub = lambda v: stats.norm.pdf(v, mu_10, sigma_10)
G_sub = lambda v: stats.norm.cdf(v, mu_10, sigma_10)

# Precompute integrals for h functions if they are often reused
h_cache = defaultdict(float)

def h_00(K, i):
    key = ("h_00", K, i)
    if key in h_cache:
        return h_cache[key]
    
    def f(x, v):
        return f_opt(x) * (K - i) * F_opt(x)**(K - i - 1) * g_sub(v) * i * G_sub(v)**(i - 1)
    
    result, _ = integrate.dblquad(f, -np.inf, np.inf, lambda v: -np.inf, lambda v: v)
    h_cache[key] = 1 - result
    return h_cache[key]

def h_01(K, i):
    key = ("h_01", K, i)
    if key in h_cache:
        return h_cache[key]
    
    def f(x, v):
        return f_sub(x) * (K - i) * F_sub(x)**(K - i - 1) * g_opt(v) * i * G_opt(v)**(i - 1)
    
    result, _ = integrate.dblquad(f, -np.inf, np.inf, lambda v: -np.inf, lambda v: v)
    h_cache[key] = 1 - result
    return h_cache[key]

def h_10(K, i):
    key = ("h_10", K, i)
    if key in h_cache:
        return h_cache[key]
    
    def f(v, x):
        return g_sub(v) * i * G_sub(v)**(i - 1) * f_opt(x) * (K - i) * F_opt(x)**(K - i - 1)
    
    result, _ = integrate.dblquad(f, -np.inf, np.inf, lambda x: -np.inf, lambda x: x)
    h_cache[key] = 1 - result
    return h_cache[key]

def h_11(K, i):
    key = ("h_11", K, i)
    if key in h_cache:
        return h_cache[key]
    
    def f(v, x):
        return g_opt(v) * i * G_opt(v)**(i - 1) * f_sub(x) * (K - i) * F_sub(x)**(K - i - 1)
    
    result, _ = integrate.dblquad(f, -np.inf, np.inf, lambda x: -np.inf, lambda x: x)
    h_cache[key] = 1 - result
    return h_cache[key]

def N(i, q_hat, gamma_hat, phi_hat, A, J, K, mean_00, mean_10):
    n1 = phi_hat * (1 - gamma_hat) * (1 - q_hat)**i * q_hat**(K - i)
    n2 = (1 - phi_hat) * gamma_hat * q_hat**i * (1 - q_hat)**(K - i)
    n3 = (1 - phi_hat) * (1 - gamma_hat) * (1 - q_hat)**i * q_hat**(K - i)
    n4 = phi_hat * gamma_hat * q_hat**i * (1 - q_hat)**(K - i)
    result = 0.0

    if A == 1 and J == 1:
        if i == 0:
            result = 0.0
        elif 0 < i < K:
            result = n1 * h_10(K, i) + n2 * h_11(K, i)
        else:  # i == K
            result = n1 + n2
    elif A == 1 and J == 0:
        if i == 0:
            result = n1 + n2
        elif 0 < i < K:
            result = n1 * h_00(K, i) + n2 * h_01(K, i)
        else:  # i == K
            result = 0.0
    elif A == 0:
        if J == 1:
            if i == 0:
                result = 0.0
            elif 0 < i < K:
                result = n3 * h_10(K, i) + n4 * h_11(K, i)
            else:  # i == K
                result = n3 + n4
        else:  # J == 0
            if i == 0:
                result = n3 + n4
            elif 0 < i < K:
                result = n3 * h_00(K, i) + n4 * h_01(K, i)
            else:  # i == K
                result = 0.0
    
    return result * (mean_00 - mean_10)

def B(i, q_hat, gamma_hat, phi_hat, A, J, K, mean_01, mean_11):
    b1 = (1 - phi_hat) * gamma_hat * (1 - q_hat)**i * q_hat**(K - i)
    b2 = phi_hat * (1 - gamma_hat) * q_hat**i * (1 - q_hat)**(K - i)
    b3 = phi_hat * gamma_hat * (1 - q_hat)**i * q_hat**(K - i)
    b4 = (1 - phi_hat) * (1 - gamma_hat) * q_hat**i * (1 - q_hat)**(K - i)
    result = 0.0

    if A == 1 and J == 1:
        if i == 0:
            result = 0.0
        elif 0 < i < K:
            result = b1 * h_10(K, i) + b2 * h_11(K, i)
        else:  # i == K
            result = b1 + b2
    elif A == 1 and J == 0:
        if i == 0:
            result = b1 + b2
        elif 0 < i < K:
            result = b1 * h_00(K, i) + b2 * h_01(K, i)
        else:  # i == K
            result = 0.0
    elif A == 0:
        if J == 1:
            if i == 0:
                result = 0.0
            elif 0 < i < K:
                result = b3 * h_10(K, i) + b4 * h_11(K, i)
            else:  # i == K
                result = b3 + b4
        else:  # J == 0
            if i == 0:
                result = b3 + b4
            elif 0 < i < K:
                result = b3 * h_00(K, i) + b4 * h_01(K, i)
            else:  # i == K
                result = 0.0
    
    return result * (mean_11 - mean_01)

def signal_ratio(s, rho_hat):
    return rho_hat / (1 - rho_hat) if s == 1 else (1 - rho_hat) / rho_hat

def choice_1(signal_ratio, N, B):
    return 0 if B == 0 else (1 if signal_ratio > N / B else 0)

def compute_choices(g_idx, genotype_matrix, Omega_matrix):
    rho_hat, q_hat, gamma_hat, phi_hat = genotype_matrix[:, g_idx]
    choice_matrix_row = np.zeros(Omega_matrix.shape[1])
    for o_idx in range(Omega_matrix.shape[1]):
        i, s, a, j = Omega_matrix[:, o_idx]
        N_value = N(i, q_hat, gamma_hat, phi_hat, a, j, 3, mu_00, mu_10)
        B_value = B(i, q_hat, gamma_hat, phi_hat, a, j, 3, mu_01, mu_11)
        signal_ratio_value = signal_ratio(s, rho_hat)
        C_value = choice_1(signal_ratio_value, N_value, B_value)
        choice_matrix_row[o_idx] = C_value
    return choice_matrix_row

def main_computation():
    K = 3
    i_values = range(K + 1)
    s_values = [0, 1]
    a_values = [0, 1]
    j_values = [0, 1]

    combinations = []
    for i in i_values:
        if i == 0:
            combinations.extend([[i, s, a, 0] for s, a in itertools.product(s_values, a_values)])
        elif i == K:
            combinations.extend([[i, s, a, 1] for s, a in itertools.product(s_values, a_values)])
        else:
            combinations.extend([[i, s, a, j] for j, s, a in itertools.product(j_values, s_values, a_values)])

    Omega_matrix = np.array(combinations).T
    print(f"Shape of the Omega matrix: {Omega_matrix.shape}")

    rho_hat_values = np.linspace(0.5001, 0.9999, 6)
    q_hat_values = np.linspace(0.0001, 0.9999, 11)
    gamma_hat_values = np.linspace(0.0001, 0.9999, 11)
    phi_hat_values = np.linspace(0.0001, 0.9999, 11)

    combinations_gene = list(itertools.product(rho_hat_values, q_hat_values, gamma_hat_values, phi_hat_values))
    genotype_matrix = np.array([list(comb) for comb in combinations_gene]).T
    print(f"Shape of the genotype matrix: {genotype_matrix.shape}")

    with mp.Pool(num_threads) as pool:
        args = [(g_idx, genotype_matrix, Omega_matrix) for g_idx in range(genotype_matrix.shape[1])]
        choice_matrix = np.array(pool.starmap(compute_choices, args))

    return choice_matrix

if __name__ == "__main__":
    print(f"Number of threads: {num_threads}")
    import time
    start_time = time.time()
    choice_matrix = main_computation()
    print(f"First 5 rows of the choice matrix:\n{choice_matrix[:5, :]}")

    # Save and load functionality
    def save_data(data, filename):
        np.save(filename, data)

    def load_data(filename):
        return np.load(filename)

    # Example usage
    choice_file = "choice_matrix.npy"
    save_data(choice_matrix, choice_file)
    loaded_matrix = load_data(choice_file)

    if loaded_matrix.size > 0:
        print("Choice matrix loaded successfully.")
        print(f"Shape of the matrix: {loaded_matrix.shape}")
    else:
        print("Choice matrix empty.")

    reduced_matrix = np.unique(loaded_matrix, axis=0)
    print(f"Shape of the reduced result matrix: {reduced_matrix.shape}")

    print(f"Execution time: {time.time() - start_time} seconds")