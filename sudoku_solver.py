"""
Basic sudoku recursive solver.
Just brute force it!
"""

def find_next_cell_to_fill(grid, i, j):
    for x in range(i,9):
        for y in range(j,9):
            if grid[x][y] == 0:
                return x,y
    for x in range(0,9):
        for y in range(0,9):
            if grid[x][y] == 0:
                return x,y
    return -1,-1

def is_valid(grid, i, j, e):
    if all([e != grid[i][x] for x in range(9)]): # Test row
        if all([e != grid[x][j] for x in range(9)]): # Test column
            # In what 3x3 cell are we?
            secTopX, secTopY = 3 *(i//3), 3 *(j//3) 
            for x in range(secTopX, secTopX+3):
                for y in range(secTopY, secTopY+3): # Test 3x3 cell
                    if grid[x][y] == e:
                        return False
            return True
    return False

def sudoku_solver(grid, i=0, j=0):
    """
    This is the recursive solving function.
    """
    i,j = find_next_cell_to_fill(grid, i, j)
    if i == -1:
        # At this point we have reached the solution
        for line in grid:
            print(line)
        return True
    for e in range(1,10):
        if is_valid(grid,i,j,e):
            grid[i][j] = e
            # Recursive call here
            if sudoku_solver(grid, i, j):
                return True
            # No solution here, execute backtracking
            grid[i][j] = 0
    # Fail, no solution at all
    return False

'''
def is_sudoku_valid(grid):
    """
    Test if the sudoku is well posed
    """
    for i in range(9):
        for j in range(9):
            if grid[i][j] != 0:
                if all([x == j or grid[i][j] != grid[i][x] for x in range(9)]): # Test row
                    if all([x == i or grid[i][j] != grid[x][j] for x in range(9)]): # Test column
                        # In what 3x3 cell are we?
                        secTopX, secTopY = 3 *(i//3), 3 *(j//3) 
                        for x in range(secTopX, secTopX+3):
                            for y in range(secTopY, secTopY+3): # Test 3x3 cell
                                if grid[x][y] == grid[i][j] and (x,y) != (i,j):
                                    return False
                    else:
                        return False
                else:
                    return False
    return True
'''