"""
The main script of the exercise H: Hidden Markov Models I.

BE3M33UI - Artificial Intelligence course, FEE CTU in Prague

Contains functions executing various inference tasks
"""

from collections import Counter
from weather import WeatherHMM
from hmm_inference import *

# Initial distribution (can be changed)
P0 = Counter({'+rain': 0.5, '-rain': 0.5})


def run_simulation(n_steps=10):
    """Simulate n_steps of WeatherHMM"""

    new_T = {
        '-rain':
            {'-rain': 0.99,
             '+rain': 0.01},
        '+rain':
            {'-rain': 0.5,
             '+rain': 0.5}
    }

    # Sensor (emission) model
    new_E = {
        '-rain':
            {'+umb': 0.9,
             '-umb': 0.1},
        '+rain':
            {'+umb': 0.9,
             '-umb': 0.1}
    }



    wtr = WeatherHMM(trans_model = new_T, emission_model = new_E)
    states, observations = wtr.simulate('+rain', n_steps)
    for state, obs in zip(states, observations):
        print(state, obs)


def run_prediction(n_steps=10):
    """Predict the belief state of WeatherHMM"""
    #prior = P0

    #wtr = WeatherHMM()
    prior = Counter({'+rain': 1})
    T = {
        '-rain':
            {'-rain': 0.9,
             '+rain': 0.1},
        '+rain':
            {'-rain': 0.2,
             '+rain': 0.8}
    }
    wtr = WeatherHMM(trans_model=T)
    print('Prediction from initial state', prior)
    p = predict(n_steps, prior, wtr)
    for i in p:
        print(i)


def run_evidence_updates():
    """Execute several evidence updates for multiple observations"""
    print('Evidence updates')
    p = Counter({'+rain': 0.5, '-rain': 0.5})
    print('Initial distribution:', p)
    wtr = WeatherHMM()
    observations = ['+umb', '+umb', '-umb', '+umb', '+umb']
    # Initialize current beliefs with uniform distribution
    for obs in observations:
        p = update_belief_by_evidence(p, obs, wtr, normalize=True)
        print(obs, p)


def run_filtering():
    """Execute forward filtering algorithm for certain evidence sequence"""
    print('Filtering')
    wtr = WeatherHMM()
    prior = Counter({'+rain': 0.5, '-rain': 0.5})
    print('Initial distribution:', prior)
    e_seq = ['+umb', '+umb', '-umb', '+umb', '+umb']
    f_seq = forward(prior, e_seq, wtr)
    for ft, et in zip(f_seq, e_seq):
        print(et, ft)


def run_likelihood():
    """Compare the likelihoods of two HMMs given the observation sequence"""
    print('Likelihood')
    wtr1 = WeatherHMM()
    E = {
        '-rain':
            {'+umb': 0.3,
             '-umb': 0.7},
        '+rain':
            {'+umb': 0.8,
             '-umb': 0.2}
    }
    wtr2 = WeatherHMM(emission_model=E)
    e_seq = ['+umb', '+umb', '-umb', '+umb', '+umb']
    prior = Counter({'+rain': 0.5, '-rain': 0.5})
    print('Likelihood of HMM1:', likelihood(prior, e_seq, wtr1))
    print('Likelihood of HMM2:', likelihood(prior, e_seq, wtr2))



if __name__=='__main__':
    #print('Comment/uncomment individual run_* functions in the main section as needed.')
    #run_simulation()
    #wtr = WeatherHMM()
    #f = Counter({'+rain': 0.5, '-rain': 0.5})
    #print(forward1(f, '+umb', wtr, normalize=True))
    #run_prediction()
    #run_evidence_updates()
    #run_filtering()
    run_likelihood()
