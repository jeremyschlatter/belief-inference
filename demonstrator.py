import torch


def policy(reward_function, transition_beliefs):
    '''
    reward_function: state, action -> real
    transition_beliefs: state, action -> Δstate
    policy result: state -> Δaction
    '''
    n_states, n_actions = reward_function.shape
    assert (n_states, n_actions, n_states) == transition_beliefs.shape

    return torch.Tensor(n_states, n_actions)


if __name__ == '__main__':
    reward_function = torch.Tensor([[-1, 1], [0, 1], [0, 2]])
    transition_beliefs = torch.Tensor([
        [[1, 0, 0], [0, 1, 0]],
        [[1, 0, 0], [0, 0, 1]],
        [[0, 1, 0], [0, 0, 1]],
    ])

    pi = policy(reward_function, transition_beliefs)
    print(pi == torch.Tensor([[0, 1], [0, 1], [0, 1]]))
