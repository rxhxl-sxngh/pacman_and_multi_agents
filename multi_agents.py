from __future__ import print_function

# multi_agents.py
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

# Rahul Singh

from builtins import range
from util import manhattan_distance
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


    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        Just like in the previous project, get_action takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (new_food) and Pacman position after moving (new_pos).
        new_scared_times holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generate_pacman_successor(action)
        new_pos = successor_game_state.get_pacman_position()
        new_food = successor_game_state.get_food()
        new_ghost_states = successor_game_state.get_ghost_states()
        new_scared_times = [ghost_state.scared_timer for ghost_state in new_ghost_states]

        "*** YOUR CODE HERE ***"
        # Convert food positions to a list using the as_list method that was used in the previous assignment (pa1)
        new_food_list = new_food.as_list()

        # Calculate the distance to the nearest food
        min_food_dist = float("inf")
        for food_pos in new_food_list:
            food_dist = manhattan_distance(new_pos, food_pos)
            min_food_dist = min(min_food_dist, food_dist)

        # Check if Pacman will collide with any ghost
        for ghost_pos in successor_game_state.get_ghost_positions():
            if manhattan_distance(new_pos, ghost_pos) < 2:
                return -float('inf')

        # Return the combined score (with the reciprocal) to minimize the distance to food
        return successor_game_state.get_score() + 1.0/min_food_dist
        # return successor_game_state.get_score()

def score_evaluation_function(current_game_state):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return current_game_state.get_score()

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

    def __init__(self, eval_fn = 'score_evaluation_function', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluation_function = util.lookup(eval_fn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def get_action(self, game_state):
        """
          Returns the minimax action from the current game_state using self.depth
          and self.evaluation_function.

          Here are some method calls that might be useful when implementing minimax.

          game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means Pacman, ghosts are >= 1

          game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action

          game_state.get_num_agents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def maximize(state, depth):
            if depth == 0 or state.is_win() or state.is_lose():
                return self.evaluation_function(state), None

            best_value = float('-inf')
            best_action = None
            legal_actions = state.get_legal_actions(0)  # Pacman's legal actions
            for action in legal_actions:
                successor_state = state.generate_successor(0, action)
                value, _ = minimize(successor_state, depth, 1)  # Call the minimizer for the next ghost
                if value > best_value:
                    best_value = value
                    best_action = action
            return best_value, best_action

        def minimize(state, depth, ghost_index):
            if depth == 0 or state.is_win() or state.is_lose():
                return self.evaluation_function(state), None

            best_value = float('inf')
            best_action = None
            legal_actions = state.get_legal_actions(ghost_index)
            for action in legal_actions:
                successor_state = state.generate_successor(ghost_index, action)
                if ghost_index == state.get_num_agents() - 1:  # Last ghost
                    value, _ = maximize(successor_state, depth - 1)  # Call the maximizer for Pacman
                else:
                    value, _ = minimize(successor_state, depth, ghost_index + 1)  # Call the minimizer for the next ghost
                if value < best_value:
                    best_value = value
                    best_action = action
            return best_value, best_action

        _, action = maximize(game_state, self.depth)
        return action
        # util.raise_not_defined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
          Returns the minimax action using self.depth and self.evaluation_function
        """
        "*** YOUR CODE HERE ***"
        return self.maximize(game_state, 0, 0, -float("inf"), float("inf"))[0]

    def alpha_beta(self, game_state, agent_Index, depth, alpha, beta):
        if depth is self.depth * game_state.get_num_agents() \
                or game_state.is_lose() or game_state.is_win():
            return self.evaluation_function(game_state)
        if agent_Index is 0:
            return self.maximize(game_state, agent_Index, depth, alpha, beta)[1]
        else:
            return self.minimize(game_state, agent_Index, depth, alpha, beta)[1]

    def maximize(self, game_state, agent_Index, depth, alpha, beta):
        best_Action = ("max", -float("inf"))
        for action in game_state.get_legal_actions(agent_Index):
            succ_Action = (action, self.alpha_beta(game_state.generate_successor(agent_Index, action),
                                      (depth + 1) % game_state.get_num_agents(), depth + 1, alpha, beta))
            best_Action = max(best_Action, succ_Action, key=lambda x: x[1])
            if best_Action[1] > beta:
                return best_Action
            else:
                alpha = max(alpha, best_Action[1])

        return best_Action

    def minimize(self, game_state, agent_Index, depth, alpha, beta):
        best_Action = ("min", float("inf"))
        for action in game_state.get_legal_actions(agent_Index):
            succ_Action = (action, self.alpha_beta(game_state.generate_successor(agent_Index, action),
                                      (depth + 1) % game_state.get_num_agents(), depth + 1, alpha, beta))
            best_Action = min(best_Action, succ_Action, key=lambda x: x[1])
            if best_Action[1] < alpha:
                return best_Action
            else:
                beta = min(beta, best_Action[1])

        return best_Action
        # util.raise_not_defined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
          Returns the expectimax action using self.depth and self.evaluation_function

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raise_not_defined()

def better_evaluation_function(current_game_state):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raise_not_defined()

# Abbreviation
better = better_evaluation_function

