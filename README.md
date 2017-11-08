### belief-inference

---

**Goal**: Build AI that can learn humans' beliefs.

**Terms**: 

- *principal*: The human (or simulated human) whose beliefs we are trying to learn.
- *agent*: The system that is trying to learn the principal's beliefs.

**Simpler problem**: Given trajectories demonstrated by the principal in an MDP and the principal's reward function, infer the state transition matrix the principal used to generate their policy.

**Approach**: Create a differentiable MDP solver. Use gradient descent to find transition matrices that maximize the likelihood of the observed trajectories, or something better than that.