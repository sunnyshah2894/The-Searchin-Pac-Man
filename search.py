# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    return generic_search(problem, util.Stack())
    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    return generic_search(problem,util.Queue())
    #util.raiseNotDefined()

def generic_search(problem,state_call_manager):

    visited = {}
    state_call_manager.push(((), problem.getStartState(),  0))  # (path,currentNode,cost)

    while not state_call_manager.isEmpty():

        current_state = state_call_manager.pop()
        current_path_directions = current_state[0]
        current_node = current_state[1]
        cost_till_now = current_state[2]

        visited[current_node] = True

        if problem.isGoalState(current_node):
            return [node[1] for node in current_path_directions]

        for successor in problem.getSuccessors(current_node):
            next_node = successor[0]
            cost_next_node = cost_till_now + successor[2]
            if not visited.has_key(next_node):
                path_to_next_node = current_path_directions + (successor,)
                state_call_manager.push((tuple(path_to_next_node), next_node, cost_next_node))

    return False

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    visited = {}
    state_call_manager = util.PriorityQueue();
    state_call_manager.push(((), problem.getStartState(),  0),0)  # (path,currentNode,cost)

    while not state_call_manager.isEmpty():

        current_state = state_call_manager.pop()
        current_path_directions = current_state[0]
        current_node = current_state[1]
        cost_till_now = current_state[2]

        visited[current_node] = True

        if problem.isGoalState(current_node):
            return [node[1] for node in current_path_directions]

        for successor in problem.getSuccessors(current_node):
            next_node = successor[0]
            cost_next_node = cost_till_now + successor[2]
            if not visited.has_key(next_node):
                path_to_next_node = current_path_directions + (successor,)
                state_call_manager.push((tuple(path_to_next_node), next_node, cost_next_node),cost_next_node)

    return False
    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    visited = {}
    state_call_manager = util.PriorityQueue();
    state_call_manager.push(((), problem.getStartState(),  0),0)  # (path,currentNode,cost)

    while not state_call_manager.isEmpty():

        current_state = state_call_manager.pop()
        current_path_directions = current_state[0]
        current_node = current_state[1]
        cost_till_now = current_state[2] - heuristic(current_node,problem)

        visited[current_node] = True

        if problem.isGoalState(current_node):
            return [node[1] for node in current_path_directions]

        for successor in problem.getSuccessors(current_node):
            next_node = successor[0]
            cost_next_node = cost_till_now + successor[2] + heuristic(next_node,problem)
            if not visited.has_key(next_node):
                path_to_next_node = current_path_directions + (successor,)
                state_call_manager.push((tuple(path_to_next_node), next_node, cost_next_node),cost_next_node)

    return False
    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
