"""
Example scripts for Robot in a maze HMM

BE3M33UI - Artificial Intelligence course, FEE CTU in Prague
"""


from hmm_inferenceM import *
from robotM import *
from utils import normalized
from matplotlib import pyplot as plt
import numpy as np
from hmmlearn import hmm


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



def test_viterbi(robot,states,obs,initial_belief):
    """Try to run Viterbi alg. for robot domain"""
    print('Running Viterbi...')
    ml_states, max_msgs = viterbi(initial_belief, obs, robot)
    for real, est in zip(states, ml_states):
        print('Real pos:', real, '| ML Estimate:', est)

if __name__=='__main__':

    # Initialization

    m1 = Maze('mazes/rect_3x2_empty.map')
    m2 = Maze('mazes/rect_5x4_empty.map')
    m3 = Maze('mazes/rect_6x10_maze.map')
    m4 = Maze('mazes/rect_6x10_obstacles.map')
    m5 = Maze('mazes/rect_8x8_maze.map')
    mazes = [m1, m2, m3, m4, m5]
    m = m4

    pos = (1,1)
    print(m)
    robot = init_maze_mod(m, pos)
    states, obs = robot.simulate(init_state=pos, n_steps=10)

    # Get list of all observations and list of all states

    all_observations = robot.get_observations()
    all_states = robot.get_states()

    # Get input sequence of observations in index form

    input_observations = []
    for k in obs:
        input_observations.append(all_observations.index(k))

    initial_belief = normalized({pos: 1 for pos in robot.get_states()})


    # Prepare model matrices

    startprob = []
    for ind,k in enumerate(initial_belief):
        startprob.append(initial_belief[k])

    trans = robot.get_transition_matrix()

    emiss = robot.get_observation_matrix()


    # Convert them into np arrays
    transmat = np.array(trans)
    emission = np.array(emiss)
    startprob = np.array(startprob)

    components = len(startprob)

    # Initialize model

    model = hmm.MultinomialHMM(n_components=components)
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.emissionprob_ = emission

    # Bob says, Alice hears analogy is used from
    # https://github.com/hmmlearn/hmmlearn/issues/70
    # Credit: ambushed

    # Reshaping sequence of indexes
    bob_says = np.array([input_observations]).T

    # Estimating position
    logprob, alice_hears = model.decode(bob_says, algorithm="viterbi")

    print("Sequence of observations:")
    for k in bob_says:
        print(all_observations[k[0]])
    print("Sequence of estimations:")
    for k in alice_hears:
        print(all_states[k])

    # Comparison to our algorithm
    test_viterbi(robot,states,obs,initial_belief)
