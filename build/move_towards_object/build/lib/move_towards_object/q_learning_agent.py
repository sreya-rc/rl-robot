import numpy as np

class QLearningAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.q_table = {}
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.action_dim = action_dim

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.array([0.5, np.random.uniform(-0.1, 0.1)])
            
        state_key = self.discretize_state(state)
        return self.q_table.get(state_key, np.array([0.5, 0.0]))

    def update_q_table(self, state, action, reward, next_state):
        state_key = self.discretize_state(state)
        next_state_key = self.discretize_state(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_dim)

        current_q = self.q_table[state_key]
        max_next_q = np.max(self.q_table[next_state_key])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key] = new_q

    def discretize_state(self, state):
        return tuple(np.round(state, decimals=1))
