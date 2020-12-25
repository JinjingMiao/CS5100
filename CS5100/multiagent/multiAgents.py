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
        distance = []
        prevFood = currentGameState.getFood()
        lowest = -float("inf")

        
        pacmanPos = list(newPos)
        pacmanX = pacmanPos[0]
        pacmanY = pacmanPos[1]

        if action == Directions.STOP:
            return lowest

        for ghostState in newGhostStates:
            if ghostState.getPosition() == tuple(pacmanPos):
                if ghostState.scaredTimer is 0:
                    return lowest 

        for food in prevFood.asList():
            x = -1*abs(food[0] - pacmanX)
            y = -1*abs(food[1] - pacmanY)
            distance.append(x+y) 

        return max(distance)

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
        self.pacmanIndex = 0


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
        actions = gameState.getLegalActions(self.index)
        highest = float("-inf")
        best_action = None
        lengthGame = gameState.getNumAgents()
        max_depth = self.depth * lengthGame
        for action in actions:
          state = gameState.generateSuccessor(self.index,action)
          score = value_mini(state,1,max_depth,self.evaluationFunction,self.index)
          if score > highest: 
            highest = score
            best_action = action
          if score <= highest:
            highest = highest
        return best_action
        
def value_mini(gameState,depth,max_depth,evaluationFunction,agentIndex):
  pacmanIndex =0
  if depth == max_depth or gameState.isWin() or gameState.isLose():
    return evaluationFunction(gameState)
  if not gameState.getLegalActions(0):
    return evaluationFunction(gameState)
  nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
  if not gameState.getLegalActions(pacmanIndex):
    return evaluationFunction(gameState)
  if nextAgentIndex != pacmanIndex:
    return minValue(gameState,depth+1,max_depth,evaluationFunction,nextAgentIndex)
  else: 
    return maxValue(gameState,depth+1,max_depth,evaluationFunction,nextAgentIndex)


def maxValue(gameState,depth,max_depth,evaluationFunction,agentIndex):
  v = float("-inf")
  actions = gameState.getLegalActions(agentIndex)
  print ("actions is" , actions)
  for action in actions:
    state = gameState.generateSuccessor(agentIndex,action)
    print ("state is" , state)
    StateValue = value_mini(state,depth,max_depth,evaluationFunction,agentIndex)
    v = max(v,StateValue)
  return v

def minValue(gameState,depth,max_depth,evaluationFunction,agentIndex):
  v = float("inf")
  actions = gameState.getLegalActions(agentIndex)
  for action in actions:
    # if action == "Stop":
    #     continue
    state = gameState.generateSuccessor(agentIndex,action)
    print ("state is" , state)
    StateValue = value_mini(state,depth,max_depth,evaluationFunction,agentIndex)
    v = min(v,StateValue)
  return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(self.index)
        highest = float("-inf")
        best_action = None
        max_depth = self.depth * gameState.getNumAgents()
        alpha = float("-inf")
        beta = float("inf")
        for action in actions:          
          state = gameState.generateSuccessor(self.index,action)
          score = value_alphabeta(state,1,max_depth,self.evaluationFunction,self.index,highest,beta)

          if score > highest: 
            highest = score
            best_action = action

        return best_action

def value_alphabeta(gameState,depth,max_depth,evaluationFunction,agentIndex,alpha,beta):
  pacmanIndex =0
  if depth == max_depth or gameState.isWin() or gameState.isLose():
    return evaluationFunction(gameState)
  if not gameState.getLegalActions(0):
    return evaluationFunction(gameState)

  nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
  if not gameState.getLegalActions(pacmanIndex):
    return evaluationFunction(gameState)
  if nextAgentIndex == pacmanIndex:
    return maxValue_alphabeta(gameState,depth+1,max_depth,evaluationFunction,nextAgentIndex,alpha,beta)
  else: 
    return minValue_alphabeta(gameState,depth+1,max_depth,evaluationFunction,nextAgentIndex,alpha,beta)


def maxValue_alphabeta(gameState,depth,max_depth,evaluationFunction,agentIndex,alpha,beta):
  v = float("-inf")
  actions = gameState.getLegalActions(agentIndex)
  for action in actions:
    state = gameState.generateSuccessor(agentIndex,action)
    stateAlpha = value_alphabeta(state,depth,max_depth,evaluationFunction,agentIndex,alpha,beta)
    print (stateAlpha)
    v = max(v,stateAlpha)
    if v > beta: 
      return v
    alpha = max(alpha,v)
  return v

def minValue_alphabeta(gameState,depth,max_depth,evaluationFunction,agentIndex,alpha,beta):
  v = float("inf")
  actions = gameState.getLegalActions(agentIndex)
  for action in actions:
    state = gameState.generateSuccessor(agentIndex,action)
    stateBeta = value_alphabeta(state,depth,max_depth,evaluationFunction,agentIndex,alpha,beta)
    v = min(v,stateBeta)
    if v < alpha: 
      return v
    beta = min(beta,v)
  return v

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
        
        highest = float("-inf")
        best_action = None
        

        actions = gameState.getLegalActions(self.index)
        for action in actions:
          state = gameState.generateSuccessor(self.index,action)
          score = value_ex(state,self.depth * gameState.getNumAgents(), 1,self.evaluationFunction,self.index)
          if score < highest: 
            highest = highest
          if score >= highest: 
            highest = score
            best_action = action


        return best_action

def value_ex(gameState,max_depth, depth,evaluationFunction,agentIndex):
  pacmanIndex = 0
  if depth == max_depth or gameState.isWin() or gameState.isLose():
    return evaluationFunction(gameState)
  if not gameState.getLegalActions(0):
    return evaluationFunction(gameState)
  nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
  if not gameState.getLegalActions(pacmanIndex):
    return evaluationFunction(gameState)
  if nextAgentIndex == pacmanIndex:
    return maxValue_ex(gameState,max_depth, depth+1,evaluationFunction,nextAgentIndex)
  else: 
    return minValue_ex(gameState,max_depth, depth+1,evaluationFunction,nextAgentIndex)


def maxValue_ex(gameState,max_depth, depth,evaluationFunction,agentIndex):

  v = float("-inf")
  actions = gameState.getLegalActions(agentIndex)
  for action in actions:
    state = gameState.generateSuccessor(agentIndex,action)
    StateValue = value_ex(state,max_depth, depth,evaluationFunction,agentIndex)
    v = max(v,StateValue)
  return v

def minValue_ex(gameState,max_depth, depth,evaluationFunction,agentIndex):
  v = set()
  actions = gameState.getLegalActions(agentIndex)
  for action in actions:
    state = gameState.generateSuccessor(agentIndex,action)
    v.add(float(value_ex(state,max_depth, depth,evaluationFunction,agentIndex)))

  return sum(v)/float(len(v))

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    distanceToFood = []
    distanceToNearestGhost = []
    distanceToCapsules =[]
    score = 0

    prevFood = currentGameState.getFood()
    currScore = currentGameState.getScore
    actions = currentGameState.getLegalActions()

    #foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsuleList = currentGameState.getCapsules()
    ans4 = len(capsuleList)
    numOfScaredGhosts = 0
    indexN = 0

    pacmanPos = list(currentGameState.getPacmanPosition())
    ans3 = currentGameState.getScore()

    for ghostState in ghostStates:
        if ghostState.scaredTimer is 0:
            numOfScaredGhosts += 1
            distanceToNearestGhost.append(indexN)
            continue


        gCoord = ghostState.getPosition()
        x = abs(gCoord[0] - pacmanPos[0])
        y = abs(gCoord[1] - pacmanPos[1])
        if (x+y) == 0:
            distanceToNearestGhost.append(index)
        else:
            distanceToNearestGhost.append(-1.0/(x+y))
            

    for food in prevFood.asList():
        distanceToFood.append(-1*(abs(food[0] - pacmanPos[0])+abs(food[1] - pacmanPos[1])))  

    if not distanceToFood:
        distanceToFood.append(indexN)
    ans1 = max(distanceToFood)
    ans2 = min(distanceToNearestGhost)
    ans5 = len(ghostStates)
    ans6= numOfScaredGhosts

    return ans1 + ans2 + ans3 - 100*ans4 - 20*(ans5 - ans6)


# Abbreviation
better = betterEvaluationFunction
