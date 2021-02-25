import numpy as np

class ExpectedSarsaAgent():
    def __init__(self, info):
        self.num_actions = info["num_actions"]
        self.num_states = info["num_states"]
        self.epsilon = info["epsilon"]
        self.step_size = info["step_size"]
        self.discount = info["discount"]
        self.rand_generator = np.random.RandomState(info["seed"])

        # Create an array for action-value estimates and initialize it to zero.
        self.q = np.zeros((self.num_states, self.num_actions))

    def agent_start(self, state):
        """The first method called when the episode starts, called after
        the environment starts.
        Args:
            state (int): the state from the
                environment's evn_start function.
        Returns:
            action (int): the first action the agent takes.
        """

        # Choose action using epsilon greedy.
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
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (int): the state from the
                environment's step based on where the agent ended up after the
                last step.
        Returns:
            action (int): the action the agent is taking.
        """

        # Choose action using epsilon greedy.
        state = 5 * state[0] + state[1]
        current_q = self.q[state, :]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions)
        else:
            action = self.argmax(current_q)

        """

        pi(any action) = epsilon / num_actions # any action might be chosen in the non-greedy case
        pi(greedy action) = pi(any action) + (1 - epsilon) / num_greedy_actions
        """
        # Perform an update (~5 lines)
        ### START CODE HERE ###
        max_q = np.max(current_q)
        num_greedy_actions = np.sum(current_q == max_q)

        non_greedy_actions_prob = (self.epsilon / self.num_actions)
        greedy_actions_prob = ((1 - self.epsilon) / num_greedy_actions) + (self.epsilon / self.num_actions)

        expected_q = 0
        for a in range(self.num_actions):
            if current_q[a] == max_q:  # This is a greedy action
                expected_q += current_q[a] * greedy_actions_prob
            else:  # This is a non-greedy action
                expected_q += current_q[a] * non_greedy_actions_prob

        self.q[self.prev_state, self.prev_action] = self.q[self.prev_state, self.prev_action] + self.step_size * (
                reward + self.discount * expected_q - self.q[self.prev_state, self.prev_action])
        ### END CODE HERE ###

        self.prev_state = state
        self.prev_action = action
        return action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        # Perform the last update in the episode (1 line)
        ### START CODE HERE ###
        self.q[self.prev_state, self.prev_action] = self.q[self.prev_state, self.prev_action] + self.step_size * (
                reward - self.q[self.prev_state, self.prev_action])
        ### END CODE HERE ###

    def argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values (Numpy array): the array of action-values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)