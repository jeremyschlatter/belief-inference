import torch


def policy(reward_function, transition_beliefs, discount):
    '''
    reward_function: state, action -> real
    transition_beliefs: state, action -> Δstate
    policy result: state -> Δaction
    '''
    n_states, n_actions = reward_function.shape
    assert (n_states, n_actions, n_states) == transition_beliefs.shape

    v = torch.Tensor(n_states)
    pi = torch.Tensor(n_states, n_actions)

    # policy iteration
    while True:
        # policy evaluation (update v)
        while True:
            delta = 0
            # TODO: vectorize
            for i in range(n_states):
                tmp = v[i]

                next_state_dist = transition_beliefs[i].t() @ pi[i]
                next_state_value = next_state_dist @ (discount * v)
                this_state_value = reward_function[i] @ pi[i]

                v[i] = this_state_value + next_state_value
                delta = max(delta, abs(tmp - v[i]))

            if delta < 0.001:
                break

        # policy improvement (update pi)
        stable = True
        # TODO: vectorize
        for i in range(n_states):
            tmp = pi[i]
            # TODO: vectorize instead of comprehension
            a_values = [
                reward_function[i, a] + discount * (transition_beliefs[i, a] @ v)
                for a in range(n_actions)
            ]
            var = torch.autograd.Variable(torch.Tensor(a_values))
            pi[i] = torch.nn.functional.softmax(var).data
            if torch.max(tmp, 0)[1][0] != torch.max(pi[i], 0)[1][0]:
                stable = False

        if stable:
            break

    return pi


if __name__ == '__main__':
    reward_function = torch.Tensor([[-1, 1], [0, 1], [0, 2]])
    transition_beliefs = torch.Tensor([
        [[1, 0, 0], [0, 1, 0]],
        [[1, 0, 0], [0, 0, 1]],
        [[0, 1, 0], [0, 0, 1]],
    ])
    discount = 0.9

    pi = policy(reward_function, transition_beliefs, discount)
    want = torch.Tensor([[0, 1], [0, 1], [0, 1]])
    print(pi == want)
    print()
    print('got:')
    print(pi)
    print('want:')
    print(want)
