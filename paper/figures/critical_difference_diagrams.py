import re
import pandas as pd
import numpy as np
import os

os.environ["PATH"] = "/Library/TeX/texbin:" + os.environ["PATH"]
from critdd import tikz_2d, tikz


def color_group_lines(tikz_code, group_colors):
    """
    Replace the default group line style with a specific color per group.

    tikz_code: string produced by tikz.to_str(...)
    group_colors: list of colors, one per group, e.g. ["blue", "red"]
    """
    lines = tikz_code.split("\n")
    colored_lines = []
    group_index = 0

    for line in lines:
        if "\\draw[group line]" in line:
            color = group_colors[group_index]
            # inject color override
            line = line.replace(
                "\\draw[group line]",
                f"\\draw[group line, draw={color}]"
            )
            group_index += 1
        colored_lines.append(line)

    return "\n".join(colored_lines)

def scale_group_spacing(tikz_code, factor=1.5):
    def repl(match):
        minrank, ypos, maxrank = match.groups()
        ypos = float(ypos) * factor
        return f"\\draw[group line] (axis cs:{minrank}, -{ypos}) -- (axis cs:{maxrank}, -{ypos});"

    pattern = r"\\draw\[group line\] \(axis cs:([\d\.]+), -([\d\.]+)\) -- \(axis cs:([\d\.]+), -([\d\.]+)\);"
    return re.sub(pattern, repl, tikz_code)

def cd_diagram_from_nemenyi(
    average_ranks,
    groups,
    treatment_names,
    *,
    reverse_x: bool = False,
    as_document: bool = False,
    tikzpicture_options: dict | None = None,
    axis_options: dict | None = None,
    preamble: str | None = None,
    title: str | None = None,
) -> str:
    """
    Build a critical difference diagram in TikZ using precomputed
    average ranks and Nemenyi groups, but keep critdd's visual style.

    Parameters
    ----------
    average_ranks : array-like of shape (k,)
        Average rank for each treatment (lower is better).
    groups : list of array-like
        Each element is a list/array of treatment indices (0-based)
        that belong to one non-significant group.
        Example: [[5, 4, 3], [3, 2, 1, 0]].
    treatment_names : list of str of length k
        Labels for the treatments in the same order as average_ranks.
    reverse_x : bool, optional
        Passed to tikz.to_str (mirror axis).
    as_document : bool, optional
        If True, wrap in a standalone LaTeX document.
    tikzpicture_options : dict, optional
        Extra options for the tikzpicture environment.
    axis_options : dict, optional
        Extra options for the axis environment.
    preamble : str, optional
        Extra LaTeX preamble when as_document=True.
    title : str, optional
        Axis title (deprecated in critdd but still supported).

    Returns
    -------
    str
        TikZ code for the CD diagram.
    """
    average_ranks = np.asarray(average_ranks, dtype=float)
    treatment_names = np.asarray(treatment_names, dtype=object)
    groups = [np.asarray(g, dtype=int) for g in groups]

    if tikzpicture_options is None:
        tikzpicture_options = {}
    if axis_options is None:
        axis_options = {}

    return tikz.to_str(
        average_ranks,
        groups,
        treatment_names,
        reverse_x=reverse_x,
        as_document=as_document,
        tikzpicture_options=tikzpicture_options,
        axis_options=axis_options,
        preamble=preamble,
        title=title,
    )

def example():
    average_ranks = [1.5, 1.5, 3.0, 4.0, 5.3, 5.8]
    treatment_names = ["m1", "m2", "m3", "m4", "m5", "m6"]
    groups = [
        [5, 4, 3],  # m6, m5, m4
        [3, 2, 1, 0],  # m4, m3, m2, m1
    ]

    tikz_code = cd_diagram_from_nemenyi(
        average_ranks,
        groups,
        treatment_names,
        reverse_x=True,
        as_document=True
    )
    group_colors = ["blue", "red"]  # in order
    tikz_code = color_group_lines(tikz_code, group_colors)
    tikz_code = scale_group_spacing(tikz_code, factor=1.7)

    with open("temp.tex", "w") as f:
        f.write(tikz_code)

def plot_all_diagrams_objectives():
    average_ranks = np.array([
        [4.0, 2.571429, 2.428571, 1.0],  # Sparsity
        [2.1428571, 1.0, 2.8571429, 4.0],  # Confidence
        [3.571428571, 3.428571429, 2.0, 1.0],  # L1 Distance
        [3.142857143, 3.714285714, 2.142857143, 1.0],  # L2 Distance
        [2.571428571, 4.0, 2.428571429, 1.0],  # DTW Distance
    ])
    groups = [
        [[0,1,2],[1,2,3]], # Sparsity
        [[2,3],[0,2],[0,1]], # Confidence
        [[0,1,2],[2,3]], # L1 Distance
        [[0,1,2],[2,3]], # L2 Distance
        [[0,1,2],[0,2,3]] # DTW Distance
    ]
    treatment_names : list[str] = ["CO_PR", "CO_SP", "CO_SP_PR", "SP_PR"]
    diagram_names : list[str] = [
        "Sparsity",
        "Confidence",
        "L1 Distance",
        "L2 Distance",
        "DTW Distance"
    ]

    tikz_code = tikz_2d.to_str(
        average_ranks,
        groups,
        treatment_names,
        diagram_names,
        reverse_x=True,
        as_document=True,
        axis_options={"title": "{Critical Difference Diagrams for Objective Combinations}"},)

    tikz.to_file("objectives_cd.pdf", tikz_code)

def plot_all_execution_times_benchmark():
    average_ranks = np.array([
        [4.5, 3.166666667, 2.333333333, 3.666666667, 1.333333333, 6.0],  # FCN
        [3.5, 3.166666667, 3.0, 4.333333333, 1.0, 6.0],  # ResNet
    ])
    groups = [
        [[0,1,3,5], [0, 1, 2 , 3], [1,2,3,4]],  # FCN
        [[0,1,2,3,5], [0,1,2,4]],  # ResNet
    ]
    treatment_names: list[str] = ["CoMTE", "$\\alpha = 0.0$", "$\\alpha = 0.5$", "$\\theta = 0.0$", "SETS", "TsEVO"]
    diagram_names: list[str] = [
        "FCN",
        "ResNet",
    ]

    tikz_code = tikz_2d.to_str(
        average_ranks,
        groups,
        treatment_names,
        diagram_names,
        reverse_x=True,
        as_document=True,
        axis_options={"title": "{Critical Difference Diagrams for Execution Times Benchmark}"}, )

    tikz.to_file("exec_times_cd.pdf", tikz_code)

def main():
    plot_all_execution_times_benchmark()

if __name__ == "__main__":
    main()
