import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


def create_function(expr, variables):
    return lambda *args: eval(expr, {
        **{var: args[i] for i, var in enumerate(variables)},
        'np': np,
        'cos': np.cos,
        'sin': np.sin,
        'tan': np.tan,
        'exp': np.exp,
        'log': np.log,
        'log10': np.log10,
        'sqrt': np.sqrt,
        'pi': np.pi,
        'acos': np.acos,
        'asin': np.asin,
        'atan': np.atan
    })

def graf(functions):
    x_range = np.linspace(-10, 10, 500)
    y_range = np.linspace(-10, 10, 500)
    X, Y = np.meshgrid(x_range, y_range)

    plt.figure(figsize=(9, 6))

    for i in range(2):
        Z = functions[i](X, Y)
        cp = plt.contour(X, Y, Z, levels=[0], colors=[f'C{i}'])
        plt.plot([], [], color=f'C{i}', label=f'f{i + 1}: {expr_array[i]} = 0')

    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.xlabel(variables[0])
    plt.ylabel(variables[1])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def printA(arr):
    for row in np.array(arr):
        if row.ndim == 0:
            print("{:f}".format(row), end=" ")
        else:  # Матрица
            for element in row:
                print("{:f}".format(element), end=" ")
            print()
    if np.array(arr).ndim == 1:
        print()

def determinant_gauss(matrix):
    n = len(matrix)
    matrix = np.array(matrix, dtype=float)
    det = 1.0
    swaps = 0

    for i in range(n):
        max_row = i
        for k in range(i + 1, n):
            if abs(matrix[k, i]) > abs(matrix[max_row, i]):
                max_row = k

        if abs(matrix[max_row, i]) < 1e-10:
            return 0.0

        if max_row != i:
            matrix[[i, max_row]] = matrix[[max_row, i]]
            swaps += 1
            det *= -1

        det *= matrix[i, i]

        for j in range(i + 1, n):
            factor = matrix[j, i] / matrix[i, i]
            matrix[j, i:] -= factor * matrix[i, i:]

    return det


def inverse_matrix_gauss(matrix):
    matrix = np.array(matrix, dtype=float)
    n = len(matrix)

    if np.abs(determinant_gauss(matrix)) < 1e-10:
        raise ValueError("Матрица вырождена")

    E = np.eye(n)
    matrix_E = np.hstack([matrix, E])

    for i in range(n):
        max_row = np.argmax(np.abs(matrix_E[i:, i])) + i
        matrix_E[[i, max_row]] = matrix_E[[max_row, i]]

        pivot = matrix_E[i, i]
        matrix_E[i, :] /= pivot

        for j in range(n):
            if j != i:
                factor = matrix_E[j, i]
                matrix_E[j, :] -= factor * matrix_E[i, :]

    return matrix_E[:, n:]


def Solvind_SoNE_Newtons_Method(expr_array, initial_approx, variables, eps=1e-6, max_iter = 100):
    symbols = [sp.Symbol(name) for name in variables]
    f_sym = [sp.sympify(expr) for expr in expr_array]
    J_sym = sp.Matrix([[sp.diff(f, var) for var in symbols] for f in f_sym])

    curr_x = np.array(initial_approx, dtype=float)

    for i in range(max_iter):
        subs_dict = dict(zip(symbols, curr_x))

        J_num = np.array(J_sym.subs(subs_dict), dtype=float)
        F_num = np.array([f.subs(subs_dict) for f in f_sym], dtype=float)

        delta_x = np.dot(inverse_matrix_gauss(J_num), -F_num)

        next_x = curr_x + delta_x
        print(f"Итерация: {i}, x = {next_x}, F(x) = {F_num}")

        if np.sqrt(np.sum((next_x - curr_x) ** 2)) < eps:
            print(f"Решение найдено за {i + 1}!")
            return next_x

        curr_x = next_x

    return curr_x


def Solvind_SoNE_Simple_Iterations_Method(expr_array, initial_approx, variables, eps=1e-6, max_iter=1000):
    n = len(variables)
    fi_expr_array = []
    for i in range(n):
        fi_expr = input(f"φ{i + 1}({', '.join(variables)}) = ")
        fi_expr_array.append(fi_expr)

    fi_funcs = [create_function(expr, variables) for expr in fi_expr_array]
    f_funcs = [create_function(expr, variables) for expr in expr_array]
    x_curr = np.array(initial_approx, dtype=float)

    for i in range(max_iter):
        x_next = np.array([fi(*x_curr) for fi in fi_funcs])
        F = np.array([f(*x_next) for f in f_funcs])
        print(f"Итерация: {i+1}, F(x): {F}")

        if np.sqrt(np.sum((x_next - x_curr)**2)) < eps:
            print(f"Решение найдено за {i + 1} итераций!")
            print(f"Невязка: {F}")
            return x_next

        x_curr = x_next

        if np.any(np.abs(x_curr) > 1e10):
            print("Метод расходится!")
            return None

    print(f"Достигнуто максимальное число итераций ({max_iter})")
    return x_curr


def check_results(expr_array, variables, solution, eps):
    if solution is None:
        print(f"Решение не найдено, проверку выполнить невозможно.")
        return

    symbols = [sp.Symbol(name) for name in variables]
    subs_dict = dict(zip(symbols, solution))

    max_residual = 0
    for i, expr in enumerate(expr_array):
        f_sym = sp.sympify(expr)
        residual = float(f_sym.subs(subs_dict).evalf())
        print(f"f{i + 1} = {residual:e}")
        max_residual = max(max_residual, abs(residual))

    if max_residual < eps:
        print(f"Решение верно: {max_residual} < {eps}")
    else:
        print(f"Решение не точное или неверное: {max_residual} > {eps}")


n = int(input("Введите кол-во уравнений: "))
eps = float(eval(input("Введите ε: ")))
#eps = 10**(-2)
variables = input("Введите переменные через пробел: ").split()

"""
cos(x-1) + y - 0.8
x - cos(y) - 2
2 1
2 + cos(y)
0.8 - cos(x-1)
"""

"""
tan(x*y) - x**2
0.8*x**2 + 2*y**2 - 1
1 1
atan(x**2) / y
sqrt(1/2 - 2/5*x**2)
-1 -1
y
-sqrt((1 - 0.8*x**2) / 2)
"""

"""
x**2 + y**2 + z**2 - 1
2*x**2 + y**2 - 4*z
3*x**2 - 4*y + z**2
0.5 0.5 0.5
sqrt(1 - y**2 - z**2)   sqrt((4*z - y**2) / 2) 
(3*x**2 + z**2) / 4     (3*x**2 + z**2) / 4
(2*x**2 + y**2) / 4     sqrt(1 - x**2 - y**2)
"""

"""
sin(x-y) - x*y + 1
x**2 - y**2 - 3/4
1 1 
sqrt(y**2 + 0.75)
(sin(x-y) + 1) / x
-1 -1
-sqrt(y**2 + 0.75)
(sin(x-y) + 1) / x
"""

"""
x**2 + y**2 - 4
3*x**2 - y
1 1
sqrt(y/3)
sqrt(4 - x**2)
-1 2
-sqrt(y/3)
sqrt(4 - x**2)
"""

expr_array, functions = [], []
for i in range(n):
    expr_array.append(input(f"f{i + 1}({', '.join(variables)}) = "))
    functions.append(create_function(expr_array[i], variables))

print("\nВведенные функции:")
for i in range(n):
    print(f"f{i + 1}: {expr_array[i]}")

if n == 2:
    graf(functions)

initial_approx = []
print("\nЗадайте начальные приближения: ")
for i in range(n):
    initial_approx.append(float(input(f"{variables[i]}(0) = ")))

print("\n===============СНАУ Методом Ньютона===============")
solution_SoNE_Newtons_Method = Solvind_SoNE_Newtons_Method(expr_array, initial_approx, variables, eps)
print(f"Решение: {solution_SoNE_Newtons_Method}")
check_results(expr_array, variables, solution_SoNE_Newtons_Method, eps)


print("\n===========СНАУ Методом Простых Итераций===========")
solution_SoNE_Simple_Iterations_Method = Solvind_SoNE_Simple_Iterations_Method(expr_array, initial_approx, variables, eps)
print(f"Решение: {solution_SoNE_Simple_Iterations_Method}")
check_results(expr_array, variables, solution_SoNE_Simple_Iterations_Method, eps)
