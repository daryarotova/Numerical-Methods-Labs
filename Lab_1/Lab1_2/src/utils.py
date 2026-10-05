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


def read_sweep_system_from_files(path_a, path_b, path_c, path_d):
    a = read_vector_from_file(path_a)
    b = read_vector_from_file(path_b)
    c = read_vector_from_file(path_c)
    d = read_vector_from_file(path_d)

    if not (len(a) == len(b) == len(c) == len(d)):
        raise ValueError(
            f'Размерности векторов не совпадают: '
            f'размерность A = {len(a)}, размерность B = {len(b)}, '
            f'размерность C = {len(c)}, размерность D = {len(d)}.')
    
    return (a, b, c, d)


if __name__ == "__main__":
    a, b, c, d = read_sweep_system_from_files(
        "data/vector_a.txt",
        "data/vector_b.txt",
        "data/vector_c.txt",
        "data/vector_d.txt",
    )
    print("a =", a)
    print("b =", b)
    print("c =", c)
    print("d =", d)
    print("shapes:", a.shape, b.shape, c.shape, d.shape)
    print("dtypes:", a.dtype, b.dtype, c.dtype, d.dtype)