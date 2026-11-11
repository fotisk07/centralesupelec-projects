import nn

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"
        self.learning_rate = 10
        self.numTrainingGames = 10000
        self.batch_size = 320
        
        # Initialize the parameters (weights and biases)
        self.parameters = []
        hidden_size1 = 400
        hidden_size2 = 400
        W1 = nn.Parameter(state_dim, hidden_size1)
        b1 = nn.Parameter(1, hidden_size1)
        W2 = nn.Parameter(hidden_size1, hidden_size2)
        b2 = nn.Parameter(1, hidden_size2)
        W3 = nn.Parameter(hidden_size2, action_dim)
        b3 = nn.Parameter(1, action_dim)
        self.set_weights([W1, b1, W2, b2, W3, b3])


    def set_weights(self, layers):
        self.parameters = []
        for i in range(len(layers)):
            self.parameters.append(layers[i])

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        Q_pred = self.run(states)
        loss = nn.SquareLoss(Q_pred, Q_target)
        return loss



    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        "*** YOUR CODE HERE ***"
        W1, b1, W2, b2, W3, b3 = self.parameters
        h1 = nn.ReLU(nn.AddBias(nn.Linear(states, W1), b1))
        h2 = nn.ReLU(nn.AddBias(nn.Linear(h1, W2), b2))
        Q_values = nn.AddBias(nn.Linear(h2, W3), b3)

        return Q_values



    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"
        loss = self.get_loss(states, Q_target)
        gradients = nn.gradients(loss, self.parameters)

        for param, grad in zip(self.parameters, gradients):
            param.update(grad, -self.learning_rate)