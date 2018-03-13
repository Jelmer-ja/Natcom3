#import pandas as pd
import numpy as np

# a 9x9 (i,j) grid
sudoku_field = []

# load the grid from the .txt field
with open('s10a.txt') as f:
    for line in f:
        int_list = [int(i) for i in line.split()]
        sudoku_field.append(int_list)

# print the field
for i, row in enumerate(sudoku_field):
    print(i, " ", row)

#max amount of pheromone that can be placed on a cell
max_pheromone = 1000

# used to store the amount of pheromone per digit/cell combination
pheromone_tabel = np.ones((9,9,9)) * max_pheromone

# TODO check if the cell has already ben filled in!!
def update_m(pher_tab, sudoku_field = sudoku_field):
    for row in range(9):
        for col in range(9):
            #TODO check if the cell has already ben filled in!!
            cell_solution = check_solutions_per_cell(sudoku_field, row, col)
            # TODO:
            for i in cell_solution:
                pher_tab[row][col][i]



def check_solutions_per_cell(field, row, col):
    solutions = []
    row_sol = check_row_wise(field,row, col)
    col_sol = check_col_wise(field, row, col)
    squ_sol = check_sub_square(field, row, col)
    solutions = row_sol + col_sol + squ_sol
    return list(set(solutions))

# check row for possible solutions
def check_row_wise(field, row_idx):
    solutions = np.arange(1,10)
    row = field[row_idx]
    for cell in row:
        if cell != 0:
            solutions = np.delete(solutions, cell-1)
    return solutions


# col for possible solutions
def check_col_wise(field, col_idx):
    solutions = np.arange(1, 10)
    col = np.ones((9,), dtype=int)
    for i, row in field:
        col[i] = row[col_idx]
    for cell in col:
        if cell != 0:
            solutions = np.delete(solutions, cell - 1)
    return solutions


# check sub-square for possible solutions:
# TODO: improve!
def check_sub_square(field, row_idx, col_idx):
    solutions = np.arange(1, 10)
    sub_square = np.ones((9,), dtype=int)
    new_row_idx, new_col_idx = 0

    if row_idx in range(0,3):
        new_row_idx = 0
    elif row_idx in range(3,6):
        new_row_idx = 3
    elif new_row_idx in range(6,9):
        new_row_idx = 6

    if col_idx in range(0, 3):
        new_col_idx = 0
    elif col_idx in range(3, 6):
        new_col_idx = 3
    elif col_idx in range(6, 9):
        new_col_idx = 6

    for i in range(row_idx, row_idx+3):
        for j in range(col_idx, col_idx+3):
            sub_square[i+j] = field[i][j]

    for cell in sub_square:
        if cell != 0:
            solutions = np.delete(solutions, cell - 1)
    return solutions