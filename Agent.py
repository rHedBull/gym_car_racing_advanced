import random

# continuous there are 3 actions :
#
# 0: steering, -1 is full left, +1 is full right
#
# 1: gas
#
# 2: breaking

class Agent:
    def __init__(self):
        self.reward = None
        self.state = None
        self.action_probabilities = {
            "none": 0.0,
        }

        self.reset()

    def reset(self):
        self.reward = 0
        self.state = "active"


    def get_action(self, agent_observation):
        # action_types = list(self.action_probabilities.keys())
        # probabilities = list(self.action_probabilities.values())
        #
        # action_type = random.choices(action_types, weights=probabilities, k=1)[0]
        return get_random_action()

    def update(self, reward):
        self.reward += reward

def get_random_action():
    return [random.uniform(-1, 1), random.uniform(0, 1), random.uniform(0, 1)]