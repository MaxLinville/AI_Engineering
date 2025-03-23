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

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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

    def evaluationFunction(self, currentGameState, action):
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

        '''
        Higher score if:
        - less food left
        - scared times increases (but don't penalize for no increase)
        - closer to food
        - eats more capsules
        '''
        capsules = successorGameState.getCapsules()
        # use diff in score as base for eval (rewards eating food, penalizes getting eaten)
        evalScore = successorGameState.getScore() - currentGameState.getScore()
        
        # avoid ghosts getting too close
        for i, ghost in enumerate(newGhostStates):
            ghostPos = ghost.getPosition()
            ghostDist = manhattanDistance(newPos, ghostPos)
            
            # if ghost not scared and very close, get out of there
            if newScaredTimes[i] <= 0 and ghostDist <= 1:
                return float('-inf')
            else:
                # just run from ghosts in general
                evalScore += ghostDist/2
        
        # more points for getting closer to food
        foodList = newFood.asList()
        if len(foodList) > 0:
            minFoodDist = float('inf')
            for food in foodList:
                foodDist = manhattanDistance(newPos, food)
                if foodDist < minFoodDist:
                    minFoodDist = foodDist
            evalScore -= minFoodDist
        
        # penalty for stopping to avoid getting stuck
        if action == Directions.STOP:
            evalScore -= 10
        
        # encourage eating more capsules
        evalScore -= len(capsules) * 10

        return evalScore
        

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        def min_value(state, agent, depth, numAgents):
            vals = []
            nextAgent = (agent + 1) % numAgents
            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                vals.append(minimaxValue(successor, nextAgent, depth, numAgents))
            v = min(vals)
            return v
        
        def max_value(state, agent, depth, numAgents):
            vals = []
            nextAgent = (agent + 1) % numAgents
            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                vals.append(minimaxValue(successor, nextAgent, depth, numAgents))
            v = max(vals)
            return v

        def minimaxValue(state, agent, depth, numAgents):
            # if state is termal state, return state's utility (evaluation function of the state)
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # with multiple ghosts, only increment depth after all ghosts have moved
            depth = depth + 1 if agent + 1 == numAgents else depth

            # if next agent is pacman (MAX), return max value
            if agent == 0:
                return max_value(state, agent, depth, numAgents)

            # if next agent is ghost (MIN), return min value
            else:
                return min_value(state, agent, depth, numAgents)

        # for each move, run the algorithm and return the best move
        numAgents = gameState.getNumAgents()
        bestValue = float('-inf')
        bestMove = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            # starts at 1 becuase we are expanding pacman's moves (agent 0)
            value = minimaxValue(successor, 1, 0, numAgents)
            if value > bestValue:
                bestValue = value
                bestMove = action
        return bestMove

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        with alpha-beta pruning
        """
        def min_value(state, agent, depth, numAgents, alpha, beta):
            # the recursive method for these is better than the iterative implementation from minimax

            # base case (no actions to take)
            if not state.getLegalActions(agent):
                return self.evaluationFunction(state)
                
            # initialize v to infinity
            v = float('inf')
            nextAgent = (agent + 1) % numAgents
            
            # check all successors and update v with the minimum value
            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                v = min(v, minimaxValue(successor, nextAgent, depth, numAgents, alpha, beta))
                
                # pruning step - if v is less than beta, min won't choose this path so ignore it
                if v < alpha:
                    return v
                    
                # update beta
                beta = min(beta, v)
                
            return v
        
        def max_value(state, agent, depth, numAgents, alpha, beta):
            if not state.getLegalActions(agent):
                return self.evaluationFunction(state)
                
            v = float('-inf')
            nextAgent = (agent + 1) % numAgents
            
            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                v = max(v, minimaxValue(successor, nextAgent, depth, numAgents, alpha, beta))
                
                # pruning step - if v is greater than beta, min won't choose this path so ignore it
                if v > beta:
                    return v
                    
                # update alpha
                alpha = max(alpha, v)
                
            return v

        # this is the exact same as minimax but with alpha and beta
        def minimaxValue(state, agent, depth, numAgents, alpha, beta):
            # if state is termal state, return state's utility (evaluation function of the state)
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # with multiple ghosts, only increment depth after all ghosts have moved
            depth = depth + 1 if agent + 1 == numAgents else depth

            # if next agent is pacman (MAX), return max value
            if agent == 0:
                return max_value(state, agent, depth, numAgents, alpha, beta)

            # if next agent is ghost (MIN), return min value
            else:
                return min_value(state, agent, depth, numAgents, alpha, beta)

        # also same as minimax but with alpha and beta
        numAgents = gameState.getNumAgents()
        bestValue = float('-inf')
        bestMove = None
        alpha = float('-inf')
        beta = float('inf')
        
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            # starts at 1 because we are expanding pacman's moves (agent 0)
            value = minimaxValue(successor, 1, 0, numAgents, alpha, beta)
            
            if value > bestValue:
                bestValue = value
                bestMove = action
                
            # another alpha update step 
            alpha = max(alpha, bestValue)
        
        return bestMove

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # identical to minimax max_value since no pruning
        def max_value(state, agent, depth, numAgents):
            vals = []
            nextAgent = (agent + 1) % numAgents
            for action in state.getLegalActions(agent):
                successor = state.generateSuccessor(agent, action)
                vals.append(expectimaxValue(successor, nextAgent, depth, numAgents))
            v = max(vals)
            return v

        def exp_value(state, agent, depth, numAgents):
            # base case
            legalActions = state.getLegalActions(agent)
            if not legalActions:
                return self.evaluationFunction(state)
                
            # initialize v to 0
            v = 0
            nextAgent = (agent + 1) % numAgents
            
            # using uniform probability for ghost moves
            probability = 1 / len(legalActions) 
            
            for action in legalActions:
                successor = state.generateSuccessor(agent, action)
                v += probability * expectimaxValue(successor, nextAgent, depth, numAgents)
                
            return v

        # again same as minimax
        def expectimaxValue(state, agent, depth, numAgents):
            # if state is termal state, return state's utility (evaluation function of the state)
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            # with multiple ghosts, only increment depth after all ghosts have moved
            depth = depth + 1 if agent + 1 == numAgents else depth

            # if next agent is pacman (MAX), return max value
            if agent == 0:
                return max_value(state, agent, depth, numAgents)

            # if next agent is ghost (EXP), return expected value
            else:
                return exp_value(state, agent, depth, numAgents)
    
        # for each move, run the algorithm and return the best move (same as minimax)
        numAgents = gameState.getNumAgents()
        bestValue = float('-inf')
        bestMove = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            # starts at 1 becuase we are expanding pacman's moves (agent 0)
            value = expectimaxValue(successor, 1, 0, numAgents)
            if value > bestValue:
                bestValue = value
                bestMove = action
        return bestMove



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: I used the exact same evaluation function as the reflex agent and modified it to 
    evaluate the current game state instead of the successor state. In summary, it rewards more food being eaten,
    capsules getting eated, being closer to food, and penalizes beign too close to ghosts unless they are 
    eatable.
    """
    "*** YOUR CODE HERE ***"
    # info about current state
    position = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    capsules = currentGameState.getCapsules()
    '''
    Higher score if:
    - less food left
    - scared times increases (but don't penalize for no increase)
    - closer to food
    '''
    # current score as base score
    evalScore = currentGameState.getScore()
    
    # avoid ghosts getting too close
    for i, ghost in enumerate(ghostStates):
        ghostPos = ghost.getPosition()
        ghostDist = manhattanDistance(position, ghostPos)
        
        # if ghost not scared and very close, get out of there
        if scaredTimes[i] <= 0 and ghostDist <= 1:
            # had to change this from float('-inf') to avoid error
            evalScore -= 1000
        else:
            # just run from ghosts in general
            evalScore += ghostDist/2

    # more points for getting closer to food
    foodList = food.asList()
    if len(foodList) > 0:
        minFoodDist = float('inf')
        for food in foodList:
            foodDist = manhattanDistance(position, food)
            if foodDist < minFoodDist:
                minFoodDist = foodDist
        evalScore -= minFoodDist
    
    # encourage eating more capsules
    evalScore -= len(capsules) * 10
    return evalScore
    

# Abbreviation
better = betterEvaluationFunction
