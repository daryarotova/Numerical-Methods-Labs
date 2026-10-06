import numpy as np


def read_vector_from_file(path):
    with open(path, "r", encoding="utf-8") as file:
        n = int(file.readline().strip())
        lines = file.readlines()
    
    tokens = []
    for line in lines:
        tokens.extend(line.strip().split())

    m = len(tokens)
    if m != n:
        raise ValueError(f'Ожидался размер {n} в файле {path}, получен {m}')
    
    numbers = [float(token) for token in tokens]
    result = np.array(numbers, dtype=float)
    return result


def read_matrix_from_file(path):
    with open(path, "r", encoding="utf-8") as file:
       n = int(file.readline().strip())
       lines = file.readlines()
    
    rows = []
    for line in lines:
        stripped = line.strip()
        if stripped == "":
            continue
        rows.append(stripped.split())

    k = len(rows)
    if k != n:
        raise ValueError(f'Ожидался размер {n} в файле {path}, получен {k}')
    for i, row in enumerate(rows):
        r = len(row)
        if r != n:
            raise ValueError(
                f'Ожидался размер {n} в файле {path} '
                f'в строке {i + 1}, получен {r}'
            )
    
    matrix = [[float(i) for i in row] for row in rows]
    result = np.array(matrix, dtype=float)
    return result


def read_epsilon_from_file(path_eps):
    with open(path_eps, "r", encoding="utf-8") as file:
        tokens = file.read().split()
    
    if len(tokens) == 0:
        raise ValueError(f'Файл {path_eps} пуст, ожидалось число epsilon')
    if len(tokens) > 1:
        raise ValueError(
            f'В файле {path_eps} ожидалось одно число, найдено {len(tokens)}: {tokens}'
        )
    
    epsilon = float(tokens[0])
    if epsilon <= 0:
        raise ValueError(f'Ожидалось число epsilon > 0, получен {epsilon} в файле {path_eps}')
    
    return epsilon


def read_iteration_system_from_files(path_A, path_b, path_eps):
    A = read_matrix_from_file(path_A)
    b = read_vector_from_file(path_b)
    eps = read_epsilon_from_file(path_eps)

    if A.shape[0] != b.shape[0]:
        raise ValueError(
            f'Размерности не совпадают: для матрицы = {A.shape[0]} в файле {path_A}, '
            f'для вектора = {b.shape[0]} в файле {path_b}'
        )
    
    return A, b, eps


if __name__ == "__main__":
    A, b, eps = read_iteration_system_from_files("data/matrix_A.txt", "data/vector_b.txt", "data/epsilon.txt")
    print("A = \n", A)
    print("b = \n", b)
    print("epsilon = ", eps)