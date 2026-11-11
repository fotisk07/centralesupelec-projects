# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # get all states
        states = self.mdp.getStates()
        # iterate through all states
        for i in range(self.iterations):
            # create a copy of the values
            values = self.values.copy()
            # iterate through all states
            for state in states:
                # if the state is terminal, the value is 0
                if self.mdp.isTerminal(state):
                    values[state] = 0
                else:
                    # get all possible actions
                    actions = self.mdp.getPossibleActions(state)
                    # create a list to store the q-values
                    q_values = []
                    # iterate through all actions
                    for action in actions:
                        # get the q-value
                        q_value = self.getQValue(state, action)
                        # append the q-value to the list
                        q_values.append(q_value)
                    # update the value of the state
                    values[state] = max(q_values)
            # update the values
            self.values = values



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # get the transition states and probabilities
        transition_states_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        # create a variable to store the q-value
        q_value = 0
        # iterate through all transition states and probabilities
        for transition_state, prob in transition_states_probs:
            # get the reward
            reward = self.mdp.getReward(state, action, transition_state)
            # get the value of the transition state
            value = self.values[transition_state]
            # update the q-value
            q_value += prob * (reward + self.discount * value)
        # return the q-value
        return q_value
    
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # get all possible actions
        actions = self.mdp.getPossibleActions(state)
        # if there are no actions, return None
        if len(actions) == 0:
            return None
        # create a variable to store the best action
        best_action = None
        # create a variable to store the best q-value
        best_q_value = float('-inf')
        # iterate through all actions
        for action in actions:
            # get the q-value
            q_value = self.getQValue(state, action)
            # if the q-value is greater than the best q-value, update the best q-value and best action
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        # return the best action
        return best_action
    

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
