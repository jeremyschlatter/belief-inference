import torch

from torch.autograd import Variable


def _to_variable(x):
    if isinstance(x, torch.Tensor):
        return Variable(x)
    assert isinstance(x, Variable)
    return x


def policy(reward_function, transition_beliefs, discount, max_iters=100):
    '''
    reward_function: state, action -> real
    transition_beliefs: state, action -> Δstate
    policy result: state -> Δaction
    '''
    # reward_function and transition_beliefs could be either Tensors or Variables.
    reward_function = _to_variable(reward_function)
    transition_beliefs = _to_variable(transition_beliefs)

    n_states, n_actions = reward_function.size()
    assert (n_states, n_actions, n_states) == transition_beliefs.size()

    v = Variable(torch.zeros(n_states), requires_grad=False)
    pi = Variable(torch.ones(n_states, n_actions) / n_actions, requires_grad=False)

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


def demonstrate(t_real, t_belief, r, discount, n, length=50):
    t_real = _to_variable(t_real)
    t_belief = _to_variable(t_belief)
    r = _to_variable(r)

    n_states, n_actions = r.size()
    assert (n_states, n_actions, n_states) == t_real.size()
    assert (n_states, n_actions, n_states) == t_belief.size()

    pi = policy(r, t_belief, discount)  # .expand(n, n_states, n_actions)
    states = Variable(torch.zeros(n).long())
    trajs = None

    for _ in range(length):
        action_dists = pi.index_select(0, states)
        actions = action_dists.multinomial(1).long()
        frame = torch.cat((states[:, None], actions), 1)[:, None, :]

        trajs = frame if trajs is None else torch.cat((trajs, frame), 1)

        state_dists = t_real.index_select(0, states).gather(1, actions.view(n, 1, 1).expand(n, 1, n_states)).squeeze(1)
        # TODO: better implementation supported by the next > 0.2.0_1 pytorch release
        # state_dists = t_real[states, actions.squeeze(1)]
        states = state_dists.multinomial(1).squeeze(1)

    return trajs


if __name__ == '__main__':
    reward_function = Variable(torch.Tensor([[-1, 1], [0, 1], [0, 2]]), requires_grad=False)
    transition_beliefs = Variable(torch.Tensor([
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
