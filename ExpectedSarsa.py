import numpy as np
"""
From Reinforcement Learning Course, University of Alberta (Coursera).
"""
class ExpectedSarsaAgent():
    def __init__(self, info):
        self.num_actions = info["num_actions"]
        self.num_states = info["num_states"]
        self.epsilon = info["epsilon"]
        self.step_size = info["step_size"]
        self.discount = info["discount"]
        self.rand_generator = np.random.RandomState(info["seed"])
        self.q = np.zeros((self.num_states, self.num_actions))

    def agent_start(self, state):
        state=5*state[0]+state[1]
        current_q = self.q[state, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        self.prev_state = state
        self.prev_action = action
        return action

    def agent_step(self, reward, state):
        state = 5 * state[0] + state[1]
        current_q = self.q[state, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        max_q = np.max(current_q)
        num_greedy_actions = np.sum(current_q == max_q)

        non_greedy_actions_prob = (self.epsilon / self.num_actions)
        greedy_actions_prob = ((1 - self.epsilon) / num_greedy_actions) + (self.epsilon / self.num_actions)

        expected_q = 0
        for a in range(self.num_actions):
            if current_q[a] == max_q:
                expected_q += current_q[a] * greedy_actions_prob
            else:
                expected_q += current_q[a] * non_greedy_actions_prob

        self.q[self.prev_state, self.prev_action] = self.q[self.prev_state, self.prev_action] + self.step_size * (
                reward + self.discount * expected_q - self.q[self.prev_state, self.prev_action])

        self.prev_state = state
        self.prev_action = action
        return action

    def agent_end(self, reward):
        self.q[self.prev_state, self.prev_action] = self.q[self.prev_state, self.prev_action] + self.step_size * (
                reward - self.q[self.prev_state, self.prev_action])

    def argmax(self, q_values):
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)