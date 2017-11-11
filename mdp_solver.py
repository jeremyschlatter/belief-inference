import torch

import torch.nn.functional as F
from torch.autograd import Variable


def _to_variable(x):
    return x if isinstance(x, Variable) else Variable(x)


def _to_tensor(x):
    return x.data if isinstance(x, Variable) else x


def gpu(x):
    return x.cuda() if torch.cuda.is_available() else x


def policy(reward_function, transition_beliefs, discount, max_iters=20):
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
    t_real = gpu(_to_tensor(t_real))
    t_belief = gpu(_to_tensor(t_belief))
    r = gpu(_to_tensor(r))

    n_states, n_actions = r.size()
    assert (n_states, n_actions, n_states) == t_real.size()
    assert (n_states, n_actions, n_states) == t_belief.size()

    pi = _to_tensor(policy(r, t_belief, discount))
    states = gpu(torch.zeros(n).long())
    trajs = None

    for _ in range(length):
        action_dists = pi.index_select(0, states)
        actions = action_dists.multinomial(1).long()
        frame = torch.cat((states[:, None], actions), 1)[:, None, :]

        trajs = frame if trajs is None else torch.cat((trajs, frame), 1)

        state_dists = t_real[states, actions.squeeze(1)]
        states = state_dists.multinomial(1).squeeze(1)

    return trajs


def avg_reward(trajs, r):
    trajs = _to_tensor(trajs)
    r = _to_tensor(r)

    trajs = trajs.view(-1, 2)
    return r[trajs[:, 0], trajs[:, 1]].mean()


def infer_belief(r, discount, trajs, initial_guess):
    assert torch.is_tensor(initial_guess)
    t_logits = Variable(gpu(initial_guess.log()), requires_grad=True)
    r = _to_variable(gpu(r))
    trajs = gpu(trajs)

    def mean_choice_log_likelihood(pi, trajs):
        choices = trajs.view(-1, 2)
        likelihoods = pi[choices.split(1, 1)]
        return likelihoods.log().mean()

    def mean_transition_log_likelihood(t, trajs):
        state_action_states = torch.cat((trajs[:, :-1, :], trajs[:, 1:, 0:1]), dim=2)
        likelihoods = t[state_action_states.view(-1, 3).split(1, 1)]
        return likelihoods.log().mean()

    def mean_entropy(ps):
        return (ps * ps.log()).sum(dim=-1).mean()

    optimizer = torch.optim.Adam([t_logits])
    for _ in range(100):
        optimizer.zero_grad()
        t_guess = F.softmax(t_logits)
        pi = policy(r, t_guess, discount)
        loss = 0

        # Increase the likelihood of demonstrator's choices.
        loss += 1 * -mean_choice_log_likelihood(pi, trajs)

        # Increase accuracy of demonstrator's beliefs.
        loss += 2 * -mean_transition_log_likelihood(t_guess, trajs)

        # Increase entropy of demonstrator's beliefs.
        loss += 1 * -mean_entropy(t_guess)

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
