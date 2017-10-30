import torch


def policy(reward_function, transition_beliefs, discount):
    '''
    reward_function: state, action -> real
    transition_beliefs: state, action -> Δstate
    policy result: state -> Δaction
    '''
    n_states, n_actions = reward_function.data.shape
    assert (n_states, n_actions, n_states) == transition_beliefs.data.shape

    v = torch.autograd.Variable(torch.zeros(n_states), requires_grad=False)
    pi = torch.autograd.Variable(torch.ones(n_states, n_actions) / n_actions, requires_grad=False)

    # policy iteration
    while True:
        # policy evaluation (update v)
        while True:
            # TODO: vectorize
            v_new = torch.autograd.Variable(torch.Tensor(n_states), requires_grad=False)
            for i in range(n_states):
                next_state_dist = transition_beliefs[i].t() @ pi[i]
                next_state_value = next_state_dist @ (discount * v)
                this_state_value = reward_function[i] @ pi[i]

                v_new[i] = this_state_value + next_state_value

            converged = torch.max(torch.abs(v - v_new)).data[0] < 0.001
            v = v_new
            if converged:
                break

        # policy improvement (update pi)
        stable = True
        # TODO: vectorize
        for i in range(n_states):
            prev_max, prev_argmax = torch.max(pi[i], 0)
            var = reward_function[i] + discount * (transition_beliefs[i] @ v)
            pi[i] = torch.nn.functional.softmax(var).data
            new_max, new_argmax = torch.max(pi[i], 0)
            if prev_argmax.data[0] != new_argmax.data[0] or abs(prev_max.data[0] - new_max.data[0]) > 0.001:
                stable = False

        if stable:
            break

    return pi


if __name__ == '__main__':
    reward_function = torch.autograd.Variable(torch.Tensor([[-1, 1], [0, 1], [0, 2]]), requires_grad=False)
    transition_beliefs = torch.autograd.Variable(torch.Tensor([
        [[1, 0, 0], [0, 1, 0]],
        [[1, 0, 0], [0, 0, 1]],
        [[0, 1, 0], [0, 0, 1]],
    ]), requires_grad=True)
    discount = 0.9

    pi = policy(reward_function, transition_beliefs, discount)
    want = torch.Tensor([[0, 1], [0, 1], [0, 1]])
    print('got:')
    print(pi)
    print('want:')
    print(want)
