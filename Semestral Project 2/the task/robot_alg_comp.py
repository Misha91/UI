"""
Example scripts for Robot in a maze HMM

BE3M33UI - Artificial Intelligence course, FEE CTU in Prague
"""


from hmm_inference import *
from robot import *
from utils import normalized
from matplotlib import pyplot as plt
import numpy as np

direction_probabilities = {
    NORTH: 0.25,
    EAST: 0.25,
    SOUTH: 0.25,
    WEST: 0.25
}

def filtering(states,obs,robot):
    """Modified filtering for robot domain"""
    initial_belief = normalized({pos: 1 for pos in robot.get_states()})
    beliefs = forward(initial_belief, obs, robot)

    # Lists of errors
    all_manhattan = []
    all_hitmiss = []

    for state, belief in zip(states, beliefs):
        #print('Real state:', state)
        #print('Sorted beliefs:')
        sort = sorted(belief.items(), key=lambda x: x[1], reverse=True)
        calculated_belief = sort[0][0]
        #print(calculated_belief)

        manhattan_distance = manhattan(state,calculated_belief)
        #returns one if "miss"
        hit_miss_error = hit_miss(state,calculated_belief)

        all_manhattan.append(manhattan_distance)
        all_hitmiss.append(hit_miss_error)

    return all_manhattan,all_hitmiss

def smoothing(states,obs,robot):
    """Modified smoothing for robot domain"""
    initial_belief = normalized({pos: 1 for pos in robot.get_states()})
    beliefs = forwardbackward(initial_belief, obs, robot)

    # Lists of errors
    all_manhattan = []
    all_hitmiss = []

    for state, belief in zip(states, beliefs):
        #print('Real state:', state)
        #print('Sorted beliefs:')

        sort = sorted(belief.items(), key=lambda x: x[1], reverse=True)
        calculated_belief = sort[0][0]
        #print(calculated_belief)

        manhattan_distance = manhattan(state, calculated_belief)
        # returns one if "miss"
        hit_miss_error = hit_miss(state, calculated_belief)

        all_manhattan.append(manhattan_distance)
        all_hitmiss.append(hit_miss_error)

    return all_manhattan,all_hitmiss

def viterbi_alg(states,obs,robot):
    """Modified Viterbi alg. for robot domain"""

    initial_belief = normalized({pos: 1 for pos in robot.get_states()})
    ml_states, max_msgs = viterbi(initial_belief, obs, robot)

    # Lists of errors
    all_manhattan = []
    all_hitmiss = []

    for real, est in zip(states, ml_states):
        #print('Real pos:', real, '| ML Estimate:', est)

        manhattan_distance = manhattan(real, est)
        # returns one if "miss"
        hit_miss_error = hit_miss(real, est)

        all_manhattan.append(manhattan_distance)
        all_hitmiss.append(hit_miss_error)

    #print(all_hitmiss)
    return all_manhattan,all_hitmiss


def init_maze_mod(m,pos):
    """Initialize robot"""
    robot = Robot(ALL_DIRS, direction_probabilities)
    #robot = Robot()
    robot.maze = m
    robot.position = pos
    #print('Robot at ', robot.position)
    return robot

def hit_miss(real, calc):
    if real != calc:
        return 1
    else: return 0


def average_error(errors):
    return sum(errors)/len(errors)

if __name__=='__main__':
    m1 = Maze('mazes/rect_3x2_empty.map')
    m2 = Maze('mazes/rect_5x4_empty.map')
    m3 = Maze('mazes/rect_6x10_maze.map')
    m4 = Maze('mazes/rect_6x10_obstacles.map')
    m5 = Maze('mazes/rect_8x8_maze.map')
    mazes = [m1, m2, m3, m4, m5]
    # Robot init position

    pos = (1,1)

    # Select number of iterations
    number_of_iter = 5
    # Select maze
    m = m1
    print(m)

    # Inititalization of error progressions
    progression_manh_distance = []
    progression_miss_percent = []

    # Running
    for n in range(number_of_iter):
                print("Iteration: ", n)
                # Initialization
                robot = init_maze_mod(m,pos)
                states, obs = robot.simulate(init_state=pos, n_steps=100)

                # Running algorithms, returning errors: manhattan, hit or miss
                fil_man, fil_hm = filtering(states,obs,robot)
                smoo_man, smoo_hm = smoothing(states, obs, robot)
                vit_man, vit_hm = viterbi_alg(states, obs, robot)

                # Getting average values
                average_manh_error = [average_error(fil_man), average_error(smoo_man),average_error(vit_man)]
                miss_percent = [average_error(fil_hm), average_error(smoo_hm),average_error(vit_hm)]

                # Displaying results
                print("Average manhattan distance from real to estimated state")
                print(average_manh_error)
                print("Percent of wrong estimations")
                print(miss_percent)

                progression_manh_distance.append(average_manh_error)
                progression_miss_percent.append(miss_percent)

    #print(progression_miss_percent)
    #print(progression_manh_distance)


    # Data modification
    progression_miss_percent = np.array(progression_miss_percent)
    progression_miss_percent = progression_miss_percent.dot(100)

    progression_manh_distance = np.array(progression_manh_distance)

    # Hit_miss plot
    plt.figure(0)
    plt.plot(range(number_of_iter),progression_miss_percent[:,0])
    plt.plot(range(number_of_iter),progression_miss_percent[:,1])
    plt.plot(range(number_of_iter),progression_miss_percent[:, 2])

    plt.legend(('Filtering', 'Smoothing', 'Viterbi'))
    plt.title("Comparison of algorithms with respect to percent of missed estimations. Maze: 1")
    plt.xlabel("[i] - Index of iteration")
    plt.ylabel("[%] - Percent")
    plt.grid()

    # Manhattan distance plot
    plt.figure(1)
    plt.plot(range(number_of_iter), progression_manh_distance[:, 0])
    plt.plot(range(number_of_iter), progression_manh_distance[:, 1])
    plt.plot(range(number_of_iter), progression_manh_distance[:, 2])

    plt.legend(('Filtering', 'Smoothing', 'Viterbi'))
    plt.title("Comparison of algorithms with respect to Manhattan underestimation. Maze: 1")
    plt.xlabel("[i] - Index of iteration")
    plt.ylabel("[cell_size] - Distance")
    plt.grid()

    # Results
    print("--- Results ---")
    print("Overall miss")
    print([average_error(progression_miss_percent[:,0]),average_error(progression_miss_percent[:,1]),average_error(progression_miss_percent[:,2])])

    print("Overall manhattan")
    print([average_error(progression_manh_distance[:,0]),average_error(progression_manh_distance[:,1]),average_error(progression_manh_distance[:,2])])

    plt.show()
