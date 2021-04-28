"""
suitable for dense graphs and it returns the minimal cost matching.
key idea:
    if a number is added to all of the entries of any row or column
    of a cost matrix, then an optimal assignment for the resulting cost
    matrix is also an optimal assignment for the original cost matrix.
Thus we can compute the maximum matching by minimizing the loss instead of
the initial weights; generating a new matrix, subtracting from the maximum entry,
the values of all the other entries.
"""


import numpy as np
from scipy.optimize import linear_sum_assignment

### STEP 1
# subtract to every element of each row the smallest value of the row
# thus every row will contain a 0
def step1(m):
    for i in range(m.shape[0]):
        m[i,:] = m[i,:] - np.min(m[i,:])

### STEP 2
# subtract to every element of each column the smallest value of the column
# thus every columns will contain a 0
def step2(m):
    for j in range(m.shape[1]):
        m[:,j] = m[:,j] - np.min(m[:,j])

### STEP 3
# find the minimal number of lines (row and columns) we have to draw in order to take all the zeros
# 1. find a start assignment covering as many tasks (y-columns) as possible
# 2. mark all rows having no assignment
# 3. mark all (unmarked) columns having zeros in newly marked row(s)
# 4. mark all rows having assignments in newly marked columns
# 5. repeat for all non-assigned rows
# 6. select marked columns and unmarked rows

def step3(m):
    dim = m.shape[0]
    assigned = np.array([])
    assignments = np.zeros(m.shape, dtype=int)

    # assigned?????
    for i in range(dim):
        for j in range(dim):
            if m[i,j]==0 and np.sum(assignments[:,j])==0 and np.sum(assignments[i,:])==0:
                assignments[i,j] = 1
                assigned = np.append(assigned, i)

    # Return evenly spaced numbers over a specified interval.
    rows = np.linspace(0, dim-1, dim).astype(int)
    marked_rows = np.setdiff1d(rows, assigned)

    # unmarked rows
    new_marked_rows = marked_rows.copy()
    marked_columns = np.array([])

    while(len(new_marked_rows)>0):
        new_marked_cols = np.array([], dtype=int)
        for nr in new_marked_rows:
            zeros_cols = np.argwhere(m[nr,:]==0).reshape(-1)
            new_marked_cols = np.append(new_marked_cols,
                                       np.setdiff1d(zeros_cols, marked_columns)
                                       )
        marked_columns = np.append(marked_columns, new_marked_cols)
        new_marked_rows = np.array([], dtype=int)


        for nc in new_marked_cols:
            new_marked_rows = np.append(new_marked_rows,
                                    np.argwhere(assignments[:,nc]==1).reshape(-1))

        marked_rows = np.unique(np.append(marked_rows, new_marked_rows))
    #unmarked_rows and marked cols
    unmarked_rows = np.setdiff1d(rows, marked_rows).astype(int)
    return unmarked_rows, np.unique(marked_columns)




### STEP 5
# find the smallest entry not covered by any line, subtract it from each row that is not crossed out,
# and then add it to each column that is crossed out
# go to ### STEP 3
def step5(m, covered_rows, covered_cols):
    uncovered_rows = np.setdiff1d(np.linspace(0, m.shape[0] - 1, m.shape[0]),
                                  covered_rows).astype(int)
    uncovered_cols = np.setdiff1d(np.linspace(0, m.shape[1] - 1, m.shape[1]),
                                  covered_cols).astype(int)

    covered_rows, covered_cols = covered_rows.astype(int), covered_cols.astype(int)
    min_val = np.max(m)

    for i in uncovered_rows:
        for j in uncovered_cols:
            if m[i,j]<min_val:
                min_val = m[i,j]

    for i in uncovered_rows:
        m[i,:] -= min_val

    for j in covered_cols:
        m[:,j] += min_val

    return m

def find_rows_single_zero(matrix):
    for i in range(matrix.shape[0]):
        if np.sum(matrix[i,:]==0)==1:
            j = np.argwhere(matrix[i,:]==0).reshape(-1)[0]
            return i,j
    return False

def find_cols_single_zero(matrix):
    for i in range(matrix.shape[1]):
        if np.sum(matrix[:, i]==0)==1:
            j = np.argwhere(matrix[:,i]==0).reshape(-1)[0]
            return i,j
    return False


def assignment_single_zero_lines(m, assignment):
    val = find_rows_single_zero(m)
    while(val):
        i, j = val[0], val[1]
        m[i,j] += 1
        m[:,j] += 1
        assignment[i,j] = 1
        val = find_rows_single_zero(m)

    val = find_cols_single_zero(m)
    while (val):
        i, j = val[0], val[1]
        m[i, :] += 1
        m[i, j] += 1
        assignment[i, j] = 1
        val = find_cols_single_zero(m)

    return assignment


def first_zero(m):
    return np.argwhere(m==0)[0][0], np.argwhere(m==0)[0][1]


# 1. Find the first row with a single 0; mark this 0 by '1' to make the assignment
# 2. Mark all the zeros in the column of the marked zero
# 3. Do the same procedure for the columns
# 4. Repeat the procedure until there are no rows and columns with single zeros
# 5. If we have more rows or columns with more than one 0, then we have to choose
#       one of the entries with value 0 and mark a cross in the cells of the remaining
#       zeros in its row and column
# 6. Repeat the procedure until thera are no unmarked 0

def final_assignment(initial_matrix, m):
    assignment = np.zeros(m.shape, dtype=int)
    assignment = assignment_single_zero_lines(m, assignment)

    while(np.sum(m==0)>0):
        i, j = first_zero(m)
        assignment[i,j] = 1
        m[i,:] += 1
        m[:,j] += 1
        assignment = assignment_single_zero_lines(m, assignment)

    return assignment*initial_matrix, assignment


def hungarian_algorithm(adj_matrix):
    m = adj_matrix.copy()
    step1(m)
    step2(m)
    # minimum number of lines to cover all the zeros
    n_lines = 0
    max_len = np.maximum(m.shape[0], m.shape[1])
    while n_lines != max_len:
        lines = step3(m)
        n_lines = len(lines[0]) + len(lines[1])
        ### STEP 4
        # check if the number of lines drawn is equal to the number of rows or column:
        # if true, we can find an optimal assignment for zeros, and the algorithm ends
        # otherwise go to ### STEP 5
        if n_lines != max_len:
            print('step5')
            step5(m, lines[0], lines[1])

    return final_assignment(adj_matrix, m)




if __name__ == '__main__':
    a = np.array([
        [102,120,152],
        [152,139,174],
        [118,146,260]
    ])
    res = hungarian_algorithm(a)
    print("\nOptimal Matching:\n", res[1], "\nValue", np.sum(res[0]))
