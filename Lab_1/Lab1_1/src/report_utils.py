def add_section(lines, title):
    lines.append(f"--- {title} ---")
    lines.append("")


def add_matrix(lines, name, matrix):
    lines.append(f"{name} =")
    lines.append(str(matrix))
    lines.append("")


def add_value(lines, name, value):
    lines.append(f"{name}: {value}")
    lines.append("")


def add_list(lines, name, items):
    lines.append(f"{name}:")
    for item in items:
        lines.append(f"  {item}")
    lines.append("")
