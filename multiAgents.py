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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        # Calculate the distance to the closest food
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        closestFoodDistance = min(foodDistances) if foodDistances else 0

        # Calculate the distance to the closest ghost
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        closestGhostDistance = min(ghostDistances) if ghostDistances else 0

        # Calculate the score based on the distances
        score = successorGameState.getScore()
        score -= closestFoodDistance
        for i, scared in enumerate(newScaredTimes):
            if scared > 0 and ghostDistances[i] == closestFoodDistance:
                score -= closestGhostDistance
                return score
        score += closestGhostDistance

        return score

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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
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
        """
        "*** YOUR CODE HERE ***"
        # Devuelve la acción óptima (primer elemento de la tupla) obtenida del resultado de maxval
        return self.maxval(gameState, 0, 0)[0]

    # Implementación de Minimax
    def minimax(self, gameState, agentIndex, depth):
        # Condiciones de terminación recursiva
        if depth is self.depth * gameState.getNumAgents() \
                or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)  # Evaluar el estado del juego

        # Turno del jugador maximizador (Pacman)
        if agentIndex == 0:
            return self.maxval(gameState, agentIndex, depth)[1]  # Llamada a maxval

        # Turno de los jugadores minimizadores (Fantasmas)
        else:
            return self.minval(gameState, agentIndex, depth)[1]  # Llamada a minval

    # Función para el jugador maximizador (Pacman)
    def maxval(self, gameState, agentIndex, depth):
        bestAction = ("max", -float("inf"))  # Inicializar la mejor acción con valor -infinito
        for action in gameState.getLegalActions(agentIndex):
            # Obtener el sucesor de gameState aplicando la acción actual
            succAction = (action, self.minimax(gameState.generateSuccessor(agentIndex, action),
                                            (depth + 1) % gameState.getNumAgents(), depth + 1))
            # Actualizar la mejor acción basada en el valor del sucesor
            bestAction = max(bestAction, succAction, key=lambda x: x[1])  # Seleccionar el máximo valor

        return bestAction  # Devolver la mejor acción y su valor asociado

    # Función para los jugadores minimizadores (Fantasmas)
    def minval(self, gameState, agentIndex, depth):
        bestAction = ("min", float("inf"))  # Inicializar la mejor acción con valor infinito
        for action in gameState.getLegalActions(agentIndex):
            # Obtener el sucesor de gameState aplicando la acción actual
            succAction = (action, self.minimax(gameState.generateSuccessor(agentIndex, action),
                                            (depth + 1) % gameState.getNumAgents(), depth + 1))
            # Actualizar la mejor acción basada en el valor del sucesor
            bestAction = min(bestAction, succAction, key=lambda x: x[1])  # Seleccionar el mínimo valor

        return bestAction  # Devolver la mejor acción y su valor asociado


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxval(gameState, 0, 0, -float("inf"), float("inf"))[0]

    # Implementación de Alpha-Beta Pruning
    def alphabeta(self, gameState, agentIndex, depth, alpha, beta):
        # Condiciones de terminación recursiva
        if depth is self.depth * gameState.getNumAgents() \
                or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)  # Evaluar el estado del juego

        # Turno del jugador maximizador (Pacman)
        if agentIndex == 0:
            return self.maxval(gameState, agentIndex, depth, alpha, beta)[1]  # Llamada a maxval con poda alfa-beta

        # Turno de los jugadores minimizadores (Fantasmas)
        else:
            return self.minval(gameState, agentIndex, depth, alpha, beta)[1]  # Llamada a minval con poda alfa-beta

    # Función para el jugador maximizador (Pacman) con Alpha-Beta Pruning
    def maxval(self, gameState, agentIndex, depth, alpha, beta):
        bestAction = ("max", -float("inf"))  # Inicializar la mejor acción con valor -infinito
        for action in gameState.getLegalActions(agentIndex):
            # Obtener el sucesor de gameState aplicando la acción actual
            succAction = (action, self.alphabeta(gameState.generateSuccessor(agentIndex, action),
                                                 (depth + 1) % gameState.getNumAgents(), depth + 1, alpha, beta))
            # Actualizar la mejor acción basada en el valor del sucesor
            bestAction = max(bestAction, succAction, key=lambda x: x[1])  # Seleccionar el máximo valor

            # Podar el árbol si el valor es mayor que beta
            if bestAction[1] > beta:
                return bestAction  # Devolver la mejor acción si se supera el límite superior beta
            else:
                alpha = max(alpha, bestAction[1])  # Actualizar alfa si no se produce la poda

        return bestAction  # Devolver la mejor acción y su valor asociado

    # Función para los jugadores minimizadores (Fantasmas) con Alpha-Beta Pruning
    def minval(self, gameState, agentIndex, depth, alpha, beta):
        bestAction = ("min", float("inf"))  # Inicializar la mejor acción con valor infinito
        for action in gameState.getLegalActions(agentIndex):
            # Obtener el sucesor de gameState aplicando la acción actual
            succAction = (action, self.alphabeta(gameState.generateSuccessor(agentIndex, action),
                                                 (depth + 1) % gameState.getNumAgents(), depth + 1, alpha, beta))
            # Actualizar la mejor acción basada en el valor del sucesor
            bestAction = min(bestAction, succAction, key=lambda x: x[1])  # Seleccionar el mínimo valor

            # Podar el árbol si el valor es menor que alpha
            if bestAction[1] < alpha:
                return bestAction  # Devolver la mejor acción si se supera el límite inferior alpha
            else:
                beta = min(beta, bestAction[1])  # Actualizar beta si no se produce la poda

        return bestAction  # Devolver la mejor acción y su valor asociado

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
