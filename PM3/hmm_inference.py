"""
Functions for inference in HMMs

BE3M33UI - Artificial Intelligence course, FEE CTU in Prague
"""

from collections import Counter
from utils import normalized


def update_belief_by_time_step(prev_B, hmm):
    """Update the distribution over states by 1 time step.

    :param prev_B: Counter, previous belief distribution over states
    :param hmm: contains the transition model hmm.pt(from,to)
    :return: Counter, current (updated) belief distribution over states
    """
    cur_B = Counter()
    states = hmm.get_states()
    # Your code here
    # print(prev_B)
    for i in states:
        for j in states:
            # print("tmp ", prev_B[states[i]], states[i], states[j], hmm.pt(states[i], states[j]))
            cur_B[j] = cur_B[j] + prev_B[i] * hmm.pt(i, j)
    # print("E ", cur_B)
    return cur_B


def predict(n_steps, prior, hmm):
    """Predict belief state n_steps to the future

    :param n_steps: number of time-step updates we shall execute
    :param prior: Counter, initial distribution over the states
    :param hmm: contains the transition model hmm.pt(from, to)
    :return: sequence of belief distributions (list of Counters),
             for each time slice one belief distribution;
             prior distribution shall not be included
    """
    B = prior  # This shall be iteratively updated
    Bs = [B]    # This shall be a collection of Bs over time steps
    # Your code here
    #raise NotImplementedError('You must implement predict()')
    for n in range(n_steps):
        B = update_belief_by_time_step(B, hmm)
        Bs.append(B)
    return Bs


def update_belief_by_evidence(prev_B, e, hmm, normalize=False):
    """Update the belief distribution over states by observation

    :param prev_B: Counter, previous belief distribution over states
    :param e: a single evidence/observation used for update
    :param hmm: HMM for which we compute the update
    :param normalize: bool, whether the result shall be normalized
    :return: Counter, current (updated) belief distribution over states
    """
    # Create a new copy of the current belief state
    cur_B = Counter(prev_B)
    states = hmm.get_states()
    for w in states:
        cur_B[w] = cur_B[w]* hmm.pe(w, e)

    if (normalize):
        cur_B = normalized(cur_B)
    # Your code here
    #raise NotImplementedError('You must implement update_belief_by_evidence()')
    return cur_B


def forward1(prev_f, cur_e, hmm, normalize=False):
    """Perform a single update of the forward message

    :param prev_f: Counter, previous belief distribution over states
    :param cur_e: a single current observation
    :param hmm: HMM, contains the transition and emission models
    :param normalize: bool, should we normalize on the fly?
    :return: Counter, current belief distribution over states
    """
    # Your code here
    #raise NotImplementedError('You must implement forward1()')
    cur_f = update_belief_by_evidence(update_belief_by_time_step(prev_f,hmm), cur_e, hmm, normalize)
    return cur_f


def forward(init_f, e_seq, hmm, normalize=True):
    """Compute the filtered belief states given the observation sequence

    :param init_f: Counter, initial belief distribution over the states
    :param e_seq: sequence of observations
    :param hmm: contains the transition and emission models
    :return: sequence of Counters, estimates of belief states for all time slices
    """
    f = init_f    # Forward message, updated each iteration
    fs = []       # Sequence of forward messages, one for each time slice
    # Your code here
    #raise NotImplementedError('You must implement forward()')
    for e in e_seq:
        f = forward1(f, e, hmm, normalize)
        #print(f)
        fs.append(f)
    return fs


def likelihood(prior, e_seq, hmm):
    """Compute the likelihood of the model wrt the evidence sequence

    In other words, compute the marginal probability of the evidence sequence.
    :param prior: Counter, initial belief distribution over states
    :param e_seq: sequence of observations
    :param hmm: contains the transition and emission models
    :return: number, likelihood
    """
    # Your code here
    #raise NotImplementedError('You must implement likelihood()')
    fs = forward(prior, e_seq, hmm, False)
    lhood = sum(fs[-1].values())

    return lhood

def backward1(next_b, next_e, hmm):
    """Propagate the backward message

    :param next_b: Counter, the backward message from the next time slice
    :param next_e: a single evidence for the next time slice
    :param hmm: HMM, contains the transition and emission models
    :return: Counter, current backward message
    """
    cur_b = Counter()
    states = hmm.get_states()
    for i in states:
        for j in states:
            cur_b[i] += hmm.pt(i,j)*hmm.pe(j, next_e)*next_b[j]
    # Your coude here
    #raise NotImplementedError('You must implement backward1()')
    return cur_b

def forwardbackward(priors, e_seq, hmm):
    """Compute the smoothed belief states given the observation sequence

    :param priors: Counter, initial belief distribution over rge states
    :param e_seq: sequence of observations
    :param hmm: HMM, contians the transition and emission models
    :return: sequence of Counters, estimates of belief states for all time slices
    """
    se = []  # Smoothed belief distributions
    fs = [priors]
    states = hmm.get_states()
    for e in e_seq:
        fs.append(forward1(fs[-1],e,hmm))

    b = Counter()
    for s in states:
        b[s] = 1

    for i in range(len(fs)-2, -1, -1):
        for s in states:
            fs[i+1][s] *= b[s]
        se.append(normalized(fs[i+1]))
        b = backward1(b, e_seq[i], hmm)
    # Your code here
    se.reverse()
    #raise NotImplementedError('You must implement forwardbackward()')
    return se

def viterbi1(prev_m, cur_e, hmm):
    """Perform a single update of the max message for Viterbi algorithm

    :param prev_m: Counter, max message from the previous time slice
    :param cur_e: current observation used for update
    :param hmm: HMM, contains transition and emission models
    :return: (cur_m, predecessors), i.e.
             Counter, an updated max message, and
             dict with the best predecessor of each state
    """
    cur_m = Counter()  # Current (updated) max message
    predecessors = {}  # The best of previous states for each current state
    states = hmm.get_states()

    for i in states:
        tmp = []
        for j in states:
            tmp.append([hmm.pt(i,j)*prev_m[j], j])
        tmp = max(tmp)
        predecessors[i] = tmp[1]
        cur_m[i] = hmm.pe(i, cur_e) * tmp[0]
    # Your code here
    #raise NotImplementedError('You must implement viterbi1()')
    return cur_m, predecessors

def viterbi(priors, e_seq, hmm):
    """Find the most likely sequence of states using Viterbi algorithm

    :param priors: Counter, prior belief distribution
    :param e_seq: sequence of observations
    :param hmm: HMM, contains the transition and emission models
    :return: (sequence of states, sequence of max messages)
    """
    ml_seq = []  # Most likely sequence of states
    ms = []  # Sequence of max messages
    # Your code here
    #raise NotImplementedError('You must implement viterbi()')
    m = forward1(priors,e_seq[0], hmm)
    ms.append(m)
   # print(m.most_common(1))
    ml_seq.append(m.most_common(1)[0][0])
    for i, e in enumerate(e_seq[1:]):
        m, p = viterbi1(ms[-1], e, hmm)
        ms.append(m)       
        ml_seq.append(m.most_common(1)[0][0])
        #print(m.most_common(1))
    return ml_seq, ms