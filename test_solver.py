import mdp_solver as mdp

import torch


def test_policy_gradient():
    T = torch.Tensor([  # noqa
        [[0.64, 0.21, 0.15], [0.5, 0.49, 0.01]],
        [[0.85, 0.01, 0.14], [0.53, 0.20, 0.27]],
        [[0.08, 0.41, 0.51], [0.33, 0.26, 0.41]],
    ])
    R = torch.Tensor([  # noqa
        [4, 4],
        [0, 0],
        [60, 60],
    ])
    discount = 0.9
    pi = mdp.policy(R, T, discount)

    # Initialize t_guess s.t. its beliefs are the reverse of T across actions.
    t_guess = torch.cat((T[:, 1:, :], T[:, 0:1, :]), 1)
    t_guess = torch.autograd.Variable(t_guess, requires_grad=True)

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


if __name__ == '__main__':
    test_policy_gradient()
