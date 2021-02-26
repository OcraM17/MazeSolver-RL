import numpy as np
"""
From Reinforcement Learning Course, University of Alberta (Coursera).
"""
class QAgent():
    def __init__(self,info):
        self.num_actions = info["num_actions"]
        self.num_states = info["num_states"]
        self.epsilon = info["epsilon"]
        self.step_size = info["step_size"]
        self.discount = info["discount"]
        self.rand_generator = np.random.RandomState(info["seed"])
        self.q = np.zeros((self.num_states, self.num_actions))

    def agent_start(self, state):
        state=10*state[0]+state[1]
        current_q = self.q[state, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)
        self.prev_state = state
        self.prev_action = action
        return action

    def agent_step(self, reward, state):
        state=10*state[0]+state[1]
        current_q = self.q[state, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)

        self.q[self.prev_state, self.prev_action] = self.q[self.prev_state, self.prev_action] + self.step_size * (
                reward + self.discount * np.max(self.q[state, :]) - self.q[self.prev_state, self.prev_action])

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
if __name__=='__main__':
    actions = []
    agent_info = {"num_actions": 4, "num_states": 25, "epsilon": 0.1, "step_size": 0.1, "discount": 1.0, "seed": 0}
    current_agent = QAgent(agent_info)
    actions.append(current_agent.agent_start(0))
    actions.append(current_agent.agent_step(2, 1))
    actions.append(current_agent.agent_step(0, 0))
    print("Action Value Estimates: \n", current_agent.q)
    print("Actions:", actions)