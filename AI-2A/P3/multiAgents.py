# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        
        "*** YOUR CODE HERE ***"
        # Get the food list
        foodList = newFood.asList()
        # Get the ghost position
        ghostPos = successorGameState.getGhostPosition(1)
        # Get the distance to the ghost
        ghostDistance = manhattanDistance(newPos, ghostPos)
        # Get the distance to the food
        foodDistance = [manhattanDistance(newPos, food) for food in foodList]
        # Get the closest food
        closestFood = min(foodDistance) if foodDistance else 0
        # Get the score
        score = successorGameState.getScore()
        # Get the number of food left
        foodLeft = len(foodList)
        # Get the number of scared ghosts
        scaredGhosts = sum([1 for ghost in newScaredTimes if ghost > 0])
        # Get the number of ghosts
        numGhosts = len(newGhostStates)
        # Get the number of capsules
        numCapsules = len(successorGameState.getCapsules())
        
        if ghostDistance < 2:
            return -999999
        
        score = 1 / (closestFood + 1) + score + 1 / (foodLeft + 1) + 1 / (numGhosts + 1) + 1 / (numCapsules + 1) + scaredGhosts

        return score



def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), None

            if agentIndex == 0:
                return max_value(state, depth, agentIndex)
            else:
                return min_value(state, depth, agentIndex)
            
        def max_value(state, depth, agentIndex):
            v = float('-inf')
            bestAction = None
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                value, _ = minimax(successor, depth, agentIndex + 1)
                if value > v:
                    v = value
                    bestAction = action
            return v, bestAction
        
        def min_value(state, depth, agentIndex):
            v = float('inf')
            bestAction = None
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == state.getNumAgents() - 1:
                    value, _ = minimax(successor, depth - 1, 0)
                else:
                    value, _ = minimax(successor, depth, agentIndex + 1)
                if value < v:
                    v = value
                    bestAction = action
            return v, bestAction
        
        _, action = minimax(gameState, self.depth, 0)
        return action
    

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alpha_beta(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), None

            if agentIndex == 0:
                return max_value(state, depth, agentIndex, alpha, beta)
            else:
                return min_value(state, depth, agentIndex, alpha, beta)
            
        def max_value(state, depth, agentIndex, alpha, beta):
            v = float('-inf')
            bestAction = None
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                value, _ = alpha_beta(successor, depth, agentIndex + 1, alpha, beta)
                if value > v:
                    v = value
                    bestAction = action
                if v > beta:
                    return v, bestAction
                alpha = max(alpha, v)
            return v, bestAction
        
        def min_value(state, depth, agentIndex, alpha, beta):
            v = float('inf')
            bestAction = None
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == state.getNumAgents() - 1:
                    value, _ = alpha_beta(successor, depth - 1, 0, alpha, beta)
                else:
                    value, _ = alpha_beta(successor, depth, agentIndex + 1, alpha, beta)
                if value < v:
                    v = value
                    bestAction = action
                if v < alpha:
                    return v, bestAction
                beta = min(beta, v)
            return v, bestAction
        
        _, action = alpha_beta(gameState, self.depth, 0, float('-inf'), float('inf'))
        return action
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state), None

            if agentIndex == 0:
                return max_value(state, depth, agentIndex)
            else:
                return exp_value(state, depth, agentIndex)
            
        def max_value(state, depth, agentIndex):
            v = float('-inf')
            bestAction = None
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                value, _ = expectimax(successor, depth, agentIndex + 1)
                if value > v:
                    v = value
                    bestAction = action
            return v, bestAction
        
        def exp_value(state, depth, agentIndex):
            v = 0
            bestAction = None
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                if agentIndex == state.getNumAgents() - 1:
                    value, _ = expectimax(successor, depth - 1, 0)
                else:
                    value, _ = expectimax(successor, depth, agentIndex + 1)
                v += value / len(state.getLegalActions(agentIndex))
            return v, bestAction
        
        _, action = expectimax(gameState, self.depth, 0)
        return action
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Get the food list
    foodList = currentGameState.getFood().asList()
    # Get the ghost position
    ghostPos = currentGameState.getGhostPosition(1)
    # Get the distance to the ghost
    ghostDistance = manhattanDistance(currentGameState.getPacmanPosition(), ghostPos)
    # Get the distance to the food
    foodDistance = [manhattanDistance(currentGameState.getPacmanPosition(), food) for food in foodList]
    # Get the closest food
    closestFood = min(foodDistance) if foodDistance else 0
    # Get the score
    score = currentGameState.getScore()
    # Get the number of food left
    foodLeft = len(foodList)
    # Get the number of scared ghosts
    scaredGhosts = sum([1 for ghost in currentGameState.getGhostStates() if ghost.scaredTimer > 0])
    # Get the number of ghosts
    numGhosts = len(currentGameState.getGhostStates())
    # Get the number of capsules
    numCapsules = len(currentGameState.getCapsules())

    if ghostDistance < 2:
        return -999999
    
    score = 1 / (closestFood + 1) + score + 1 / (foodLeft + 1) + 1 / (numGhosts + 1) + 1 / (numCapsules + 1) + scaredGhosts

    return score

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
