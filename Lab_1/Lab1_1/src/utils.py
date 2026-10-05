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
    with open (path, "r", encoding="utf-8") as file:
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


def read_system_from_files(path_A, path_b):
    A = read_matrix_from_file(path_A)
    b = read_vector_from_file(path_b)

    if A.shape[0] != b.shape[0]:
        raise ValueError(
            f'Размерности не совпадают: для матрицы = {A.shape[0]} в файле {path_A}, '
            f'для вектора = {b.shape[0]} в файле {path_b}'
        )
    return A, b

if __name__ == "__main__":
    A, b = read_system_from_files("data/matrix_A.txt", "data/vector_b.txt")
    print("A = \n", A)
    print("b =", b)