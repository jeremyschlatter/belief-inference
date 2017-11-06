import torch


def policy(reward_function, transition_beliefs, discount, max_iters=100):
    '''
    reward_function: state, action -> real
    transition_beliefs: state, action -> Δstate
    policy result: state -> Δaction
    '''
    # reward_function and transition_beliefs could be either Tensors or Variables.
    if isinstance(reward_function, torch.Tensor):
        reward_function = torch.autograd.Variable(reward_function)
    if isinstance(transition_beliefs, torch.Tensor):
        transition_beliefs = torch.autograd.Variable(transition_beliefs)

    n_states, n_actions = reward_function.data.shape
    assert (n_states, n_actions, n_states) == transition_beliefs.data.shape

    v = torch.autograd.Variable(torch.zeros(n_states), requires_grad=False)
    pi = torch.autograd.Variable(torch.ones(n_states, n_actions) / n_actions, requires_grad=False)

    # policy iteration
    for i in range(max_iters):
        # policy evaluation (update v)
        for j in range(max_iters):
            v_new = (transition_beliefs * pi.unsqueeze(-1)).sum(dim=1) @ (discount * v) + (reward_function * pi).sum(dim=1)
            converged = torch.max(torch.abs(v - v_new)).data[0] < 0.001
            v = v_new
            if converged:
                break

        # policy improvement (update pi)
        pi_new = torch.nn.functional.softmax(reward_function + discount * (transition_beliefs @ v))
        converged = torch.max(torch.abs(pi - pi_new)).data[0] < 0.001
        pi = pi_new
        if converged:
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
