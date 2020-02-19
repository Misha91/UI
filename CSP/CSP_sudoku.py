"""
B(E)3M33UI - Artificial Intelligence course, FEE CTU in Prague
Jiri Spilka, CVUT, Praha 2018
"""
import copy

from sudoku_tasks import task_easy, task_easy_solution, task_hard, \
    task_hard_solution, task_hardest, \
    task_hardest_solution


def solve(sudoku):

    solved_sudoku = []
    #test([[0,0,0], [0,0,0], [0,0,0]])
    #test([0,0,0])
    tmp, solved_sudoku = solver(sudoku[:], 0)
    for l in solved_sudoku:
        print(l)
    return solved_sudoku





def solver(sud, time):
    #print(sud[0])
    for l in range(len(sud)):
        for c in range(len(sud[l])):
            if sud[l][c] == 0:
                for k in range(1,10):
                   # if time==0: print(l, c, k, sud[l])
                    if checker(l, c, k, sud[:]):
                        sud2 = copy.copy(sud)
                        sud2[l][c] = k

                        res, tmp = solver(copy.copy(sud2), 5)
                        if (res == True): return True, tmp
                        sud[l][c] = 0
                        del sud2
                if (sum(sud[l])!=45): return False, []
    for l in range(len(sud)):
        for c in range(len(sud[l])):
            if sud[l][c] == 0: return False, []

    return True, sud

def checker(l, c, val, sudoku):
    for k in sudoku[l]:
        if k == val: return False
    for k in range(0,9):
        if sudoku[k][c] == val: return False
    i, j = l//3, c//3

    for k in range(i*3, i*3+3):
        for l in range(j*3, j*3 + 3):
            #print(k, l)
            if sudoku[k][l] == val: return False
    return True


def check_solution(sudoku_solved, sudoku_solution):

    for i in range(9):
        for j in range(9):
            print(sudoku_solved[i][j], sudoku_solution[i][j])
            if sudoku_solved[i][j] != sudoku_solution[i][j]:
                print('Not equal', sudoku_solved[i][j], sudoku_solution[i][j])
                return False
    return True


if __name__ == '__main__':

    solution = solve(task_easy)
    assert check_solution(solution, task_easy_solution)

    solution = solve(task_hard)
    assert check_solution(solution, task_hard_solution)

    solution = solve(task_hardest)
    assert check_solution(solution, task_hardest_solution)
