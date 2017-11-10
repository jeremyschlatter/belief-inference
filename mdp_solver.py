import torch

import torch.nn.functional as F
from torch.autograd import Variable


def _to_variable(x):
    return x if isinstance(x, Variable) else Variable(x)


def gpu(x):
    return x.cuda() if torch.cuda.is_available() else x


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

    if False:
        return policy_iteration(reward_function, transition_beliefs, discount, max_iters, n_states, n_actions)
    else:
        return value_iteration(reward_function, transition_beliefs, discount, max_iters, n_states, n_actions)


def policy_iteration(reward_function, transition_beliefs, discount, max_iters, n_states, n_actions):
    v = Variable(gpu(torch.zeros(n_states)), requires_grad=False)
    pi = Variable(gpu(torch.ones(n_states, n_actions) / n_actions), requires_grad=False)

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
        pi_new = F.softmax(reward_function + discount * (transition_beliefs @ v))
        converged = torch.max(torch.abs(pi - pi_new)).data[0] < 0.001
        pi = pi_new
        if converged:
            break

    return pi


def value_iteration(reward_function, transition_beliefs, discount, max_iters, n_states, n_actions):
    v = Variable(gpu(torch.zeros(n_states)), requires_grad=False)

    for i in range(max_iters):
        # s x a x s' values
        v_new = transition_beliefs * (reward_function[:, :, None] + discount * v[None, None, :])
        # sum over future states
        v_new = v_new.sum(dim=2)
        # softmax over actions
        v_new = (F.softmax(v_new) * v_new).sum(dim=1)
        # v_new = v_new.exp().sum(dim=1).log()

        converged = torch.max(torch.abs(v - v_new)).data[0] < 0.001
        v = v_new
        if converged:
            break

    return F.softmax(reward_function + discount * (transition_beliefs @ v))


def demonstrate(t_real, t_belief, r, discount, n, length=50):
    t_real = gpu(_to_variable(t_real))
    t_belief = gpu(_to_variable(t_belief))
    r = gpu(_to_variable(r))

    n_states, n_actions = r.size()
    assert (n_states, n_actions, n_states) == t_real.size()
    assert (n_states, n_actions, n_states) == t_belief.size()

    pi = policy(r, t_belief, discount)  # .expand(n, n_states, n_actions)
    states = Variable(gpu(torch.zeros(n).long()))
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


def mean_choice_log_likelihood(pi, trajs):
    choices = trajs.view(-1, 2)
    likelihoods = pi.index_select(0, choices[:, 0]).gather(1, choices[:, 1:]).squeeze(1)
    # TODO: better implementation supported by the next > 0.2.0_1 pytorch release
    # likelihoods = pi[choices[:, 0], choices[:, 1]]
    return likelihoods.log().mean()


def infer_belief(t_real, r, discount, trajs, initial_guess=None):
    if initial_guess is None:
        initial_guess = torch.rand(*t_real.size())

    t_logits = Variable(gpu(initial_guess.data.log()), requires_grad=True)
    t_real = gpu(t_real)
    r = gpu(r)
    trajs = gpu(trajs)

    optimizer = torch.optim.Adam([t_logits])
    for _ in range(200):
        optimizer.zero_grad()
        t_guess = F.softmax(t_logits)
        pi = policy(r, t_guess, discount)
        loss = -mean_choice_log_likelihood(pi, trajs)
        loss.backward()
        optimizer.step()

    return t_guess


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
