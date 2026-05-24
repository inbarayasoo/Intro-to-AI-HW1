import numpy as np
from HaifaEnv import HaifaEnv
from typing import List, Tuple
import heapdict

C_PASSWAY = 100


def h_haifa(env: HaifaEnv, state: int):
    state_rc = env.to_row_col(state)

    goal_states = env.get_goal_states()
    manhattan_distance_list = []
    for g in goal_states:
        g_rc = env.to_row_col(g)
        manhattan_distance = np.linalg.norm(np.array(state_rc) - np.array(g_rc), ord=1)
        manhattan_distance_list = manhattan_distance_list + [manhattan_distance]

    return min(min(manhattan_distance_list), C_PASSWAY)


class Node:
    def __init__(self, state: int, path: List[int], cost: float):
        self.state = state
        self.path = path
        self.cost = cost


class BFSGAgent:
    def __init__(self) -> None:
        pass

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        # init open and closed
        open_list = np.array([], dtype=object)
        closed = set()

        # create the start node
        start_state = env.get_initial_state()
        node = Node(start_state, [], 0)

        if env.is_final_state(node.state):
            return node.path, node.cost, len(closed)

        open_list = np.append(open_list, node)

        while open_list.size != 0:
            # pop from open list
            node = open_list[0]
            open_list = open_list[1:]
            # add do closed list
            closed.add(node.state)

            # create and add successors to open list
            for action, successor in env.succ(node.state).items():
                new_state, step_cost, terminated = successor
                if new_state is None:
                    break

                child = Node(new_state, node.path + [action], node.cost + step_cost)

                if new_state not in closed and all(n.state != new_state for n in open_list):
                    if env.is_final_state(child.state):
                        return child.path, child.cost, len(closed)
                    open_list = np.append(open_list, child)
        raise NotImplementedError


class GreedyAgent:
    def __init__(self) -> None:
        pass

    def search(self, env: HaifaEnv) -> Tuple[List[int], float, int]:
        # init open and closed
        open_list = heapdict.heapdict()
        open_nodes = {}
        closed = set()

        # create the start node
        start_state = env.get_initial_state()
        start_node = Node(start_state, [], 0)

        open_list[start_node.state] = (h_haifa(env, start_node.state), start_node.state)
        open_nodes[start_node.state] = start_node

        while len(open_list) != 0:
            # pop from open list
            node_state, (f_val, state) = open_list.popitem()
            node = open_nodes[node_state]
            del open_nodes[node.state]

            if env.is_final_state(node.state):
                return node.path, node.cost, len(closed)

            # add do closed list
            closed.add(node.state)

            # create and add successors to open list
            for action, successor in env.succ(node.state).items():
                new_state, step_cost, terminated = successor
                if new_state is None:
                    break

                if new_state not in open_nodes and new_state not in closed:
                    child = Node(new_state, node.path + [action], node.cost + step_cost)
                    open_nodes[child.state] = child
                    open_list[child.state] = (h_haifa(env, child.state), child.state)
        raise NotImplementedError


class AStarEpsilonAgent:
    def __init__(self):
        pass
        
    def h_focal(self, env: HaifaEnv, state: int) -> float: # heuristic for focal list (you don't have to use it)
        return h_haifa(env, state)

    def search(self, env: HaifaEnv, epsilon: float = None):
        # init open and closed
        open_list = {}
        nodes = {}
        closed = set()
        removed_from_closed = 0

        # create the start node
        start_state = env.get_initial_state()
        start_node = Node(start_state, [], 0)

        open_list[start_node.state] = h_haifa(env, start_node.state)
        nodes[start_node.state] = start_node

        while len(open_list) != 0:
            # create focal list
            min_f_val = min(open_list.values())
            threshold = (1 + epsilon) * min_f_val
            focal = heapdict.heapdict()
            for state in open_list:
                if open_list[state] <= threshold:
                    focal[state] = (self.h_focal(env, state), open_list[state], state)

            # pop from focal list and delete from open list
            node_state, (h_focal_val, h_focal_f, h_focal_state) = focal.popitem()
            del open_list[node_state]
            node = nodes[node_state]

            if env.is_final_state(node_state):
                return node.path, node.cost, len(closed) + removed_from_closed

            # add do closed list
            closed.add(node_state)

            # create and add successors to open list
            for action, successor in env.succ(node.state).items():
                new_state, step_cost, terminated = successor
                if new_state is None:
                    break

                new_cost = node.cost + step_cost
                new_path = node.path + [action]
                # not in open or closed
                if new_state not in nodes:
                    child = Node(new_state, new_path, new_cost)
                    open_list[child.state] = h_haifa(env, child.state) + child.cost
                    nodes[child.state] = child
                # in open or closed and with worst g(n)
                elif nodes[new_state].cost > new_cost:
                    nodes[new_state].cost = new_cost
                    nodes[new_state].path = new_path
                    open_list[new_state] = h_haifa(env, new_state) + new_cost
                    if new_state in closed:
                        closed.remove(new_state)
                        removed_from_closed = removed_from_closed + 1
        raise NotImplementedError


class AStarAgent(AStarEpsilonAgent):
    def __init__(self) -> None:
        super().__init__()

    def search(self, env: HaifaEnv, epsilon=0) -> Tuple[List[int], float, int]:
        epsilon = 0
        return super().search(env, epsilon)



