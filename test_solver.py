from functools import wraps

import mdp_solver as mdp

import torch
from torch.autograd import Variable


def random_transitions(s, a):
    t = torch.rand(s, a, s)
    return t / t.sum(dim=2)[:, :, None]


def manual_seed(f):
    @wraps(f)
    def wrapper():
        # Derive the seed from the test name.
        #
        # This ensures seeds are the same between test runs (essential for
        # reproducibility), but also different in most tests (lets us explore
        # seed space a little more).
        torch.manual_seed(sum(ord(c) for c in f.__name__))
        f()
    return wrapper


@manual_seed
def test_policy_gradient():
    '''We should be able to do gradient descent on T -> pi'''
    T = random_transitions(3, 2)  # noqa
    R = torch.Tensor([  # noqa
        [4, 4],
        [0, 0],
        [60, 60],
    ])
    discount = 0.9
    pi = mdp.policy(R, T, discount)

    # Initialize t_guess s.t. its beliefs are the reverse of T across actions.
    t_guess = torch.cat((T[:, 1:, :], T[:, 0:1, :]), 1)
    t_guess = Variable(t_guess, requires_grad=True)

    optimizer = torch.optim.Adam([t_guess])
    for i in range(100):
        optimizer.zero_grad()
        pi_guess = mdp.policy(R, t_guess, discount)
        loss = ((pi - pi_guess) ** 2).sum()
        if i == 0:
            assert loss.data[0] > 5
        loss.backward()
        optimizer.step()

    print('pi:', pi)
    print('pi_guess:', pi_guess)
    print('loss:', loss.data[0])
    print('T:', T)
    print('t_guess:', t_guess)
    assert loss.data[0] < 0.00001


@manual_seed
def test_belief_inference():
    s, a = 20, 3
    t_real = Variable(random_transitions(s, a))
    t_belief = Variable(random_transitions(s, a))
    R = torch.rand(s, a) * 10  # noqa
    discount = 0.9
    trajs = mdp.demonstrate(t_real, t_belief, R, discount, 1000)

    def loss(guess):
        return ((t_real - t_guess) ** 2).sum().data[0]
    t_guess = Variable(random_transitions(s, a), requires_grad=True)
    initial_loss = loss(t_guess)

    t_guess = mdp.infer_belief(t_real, R, discount, trajs, initial_guess=t_guess)
    final_loss = loss(t_guess)

    print(f'initial loss: {initial_loss}\nfinal loss: {final_loss}')
    assert final_loss < initial_loss


if __name__ == '__main__':
    test_belief_inference()
