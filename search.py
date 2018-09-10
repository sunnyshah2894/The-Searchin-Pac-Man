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


def findShortestPathUsingDFS(problem, current, visited, current_path_directions, current_move, direction_types):
    if problem.isGoalState(current):
        current_path_directions = current_path_directions + (current_move,)
        result_till_now = [direct for direct in current_path_directions]
        return result_till_now
    else:
        visited[current] = True
        current_path_directions = current_path_directions + (current_move,)
        for prob in problem.getSuccessors(current):
            if not visited.has_key(prob[0]):
                result = findShortestPathUsingDFS(problem, prob[0], visited, current_path_directions, direction_types.get(prob[1]), direction_types)
                if result:
                    return result
    return None


def find_path_using_dfs(problem, current, current_path_directions, direction_types, visited):

    if problem.isGoalState(current):
        return current_path_directions
    else:
        visited[current] = True
        for prob in problem.getSuccessors(current):
            next_node = prob[0]
            direction_to_take = (direction_types.get(prob[1]),)
            path_to_next_node = (current_path_directions + direction_to_take)
            if not visited.has_key(prob[0]):
                result = find_path_using_dfs(problem, next_node, path_to_next_node, direction_types, visited)
                if result:
                    return result
    return None

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    from game import Directions
    direction_types = {
        "South" : Directions.SOUTH,
        "West" : Directions.WEST,
        "North" : Directions.NORTH,
        "East" : Directions.EAST
    }

    visited = {}
    path_to_goal = ()
    myStack = util.Stack()
    result = None
    myStack.push(((),problem.getStartState()))

    while( not myStack.isEmpty() ):
        currentNode = myStack.pop()
        print "working on ",currentNode
        current_path_directions = currentNode[0]
        current = currentNode[1]
        visited[current] = True
        if problem.isGoalState(current):
            result = current_path_directions
            return result

        for prob in problem.getSuccessors(current):
            next_node = prob[0]
            direction_to_take = (direction_types.get(prob[1]),)
            path_to_next_node = (current_path_directions + direction_to_take)
            if not visited.has_key(prob[0]):
                print "pushing ",next_node
                myStack.push((path_to_next_node,next_node))
                #result = find_path_using_dfs(problem, next_node, path_to_next_node, direction_types, visited)
                if result:
                    return result
    return []

    # visited[problem.getStartState()] = True
    # for prob in problem.getSuccessors(problem.getStartState()):
    #     result = findShortestPathUsingDFS(problem,prob[0],visited,directions,direction.get(prob[1]),direction)
    #     if result:
    #         return result

    """
    result = find_path_using_dfs(problem, problem.getStartState(), path_to_goal, direction_types, visited)
    if result:
        return result

    return None
    """

    #return min(result, key=len)
    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
