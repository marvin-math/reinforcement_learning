from operator import truediv
from os import rename
import random
import numpy as np
import copy
import heapq

from numpy.core.defchararray import center
from numpy.lib.utils import _median_nancheck


#
# Possible actions
#
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
BOOST_LEFT = 4
BOOST_DOWN = 5
BOOST_RIGHT = 6
BOOST_UP = 7

REGULAR_ACTIONS = [ LEFT, DOWN, RIGHT, UP ]
BOOST_ACTIONS = [ BOOST_LEFT, BOOST_DOWN, BOOST_RIGHT, BOOST_UP ]
ACTIONS = REGULAR_ACTIONS + BOOST_ACTIONS

BOOST_TO_DIRECTION = {
    BOOST_LEFT: LEFT,
    BOOST_DOWN: DOWN,
    BOOST_RIGHT: RIGHT,
    BOOST_UP: UP,
}

ACTIONS_DESCRIPTION = [b'l', b'd', b'r', b'u', b'L', b'D', b'R', b'U']

#
# Nomenclature:
# 'S' = starting point
# 'G' = goal point
# 'A' = asteroid
# '-' = empty space
# 'x' = pseudo-reward tile
#
MAPS = {
    "6x6": [
        "AA-A-G",
        "--A---",
        "----AA",
        "--A--A",
        "A--A--",
        "S-----",
    ],
    "6x6_optimal_pseudo_reward": [
        "AA-A8G",
        "-2A678",
        "-345AA",
        "--A--A",
        "A--A--",
        "S1----",
    ],
    "6x6_approximate_pseudo_reward": [
        "AA-A8G",
        "--A---",
        "-3-5AA",
        "--A--A",
        "A--A--",
        "S1----",
    ],
     "6x6_unvalid": [
        "AA-A-G",
        "--A---",
        "----AA",
        "--A--A",
        "AAAA--",
        "S--A--",
    ],
}


def next_pos(grid_map, row, col, action):
  nrow, ncol = grid_map.shape
  def next_square(row, col, action):
    if action == LEFT:
        col = max(col - 1, 0)
    elif action == DOWN:
        row = min(row + 1, nrow - 1)
    elif action == RIGHT:
        col = min(col + 1, ncol - 1)
    elif action == UP:
        row = max(row - 1, 0)
    return (row, col)

  row_prev = copy.copy(row)
  col_prev = copy.copy(col)

  path = []
  if action in REGULAR_ACTIONS:
    path = [next_square(row, col, action)]
  elif action in BOOST_ACTIONS:
    action_direction = BOOST_TO_DIRECTION[action]
    path = []
    for _ in range(4):
      (row, col) = next_square(row, col, action_direction)
      path.append((row, col))
  
  crashed_into_astroid = False
  r_new, c_new = path[-1]
  for square in path:
      r, c = square
      letter = grid_map[r, c]
      if bytes(letter) == b"A":
          crashed_into_astroid = True
          r_new, c_new = row_prev, col_prev
          break

  return crashed_into_astroid, r_new, c_new

# Dijkstra to check that it's a valid path + return shortest path
def dijkstra(grid_map):
  nrow, ncol = grid_map.shape
  value_map = np.zeros((nrow, ncol), dtype=[('val', int),('row', int), ('col', int)])
  value_map.fill((0, -1, -1))

  frontier, discovered = [], set()
  goal = np.where(grid_map == b'G')
  goal_row, goal_col = int(goal[0]), int(goal[1])
  frontier.append((10, goal_row, goal_col))
  value_map[goal_row][goal_col] = (10, -1, -1)

  heapq.heapify(frontier)
  ACTIONS_DESCRIPTION = [b'l', b'd', b'r', b'u', b'L', b'D', b'R', b'U']

  # frontier[] tracks the path from the earliest fully discovered tile
  # if frontier[] becomes empty then we have exhausted all valid paths
  valid = False
  while frontier:
    val, r, c = heapq.heappop(frontier)
    if not (r, c) in discovered:
        discovered.add((r, c))
        for action in ACTIONS:
          crashed_into_astroid, r_new, c_new = next_pos(grid_map, r, c, action)
          if crashed_into_astroid:
            continue
          if (r_new, c_new) in discovered:
            continue
          if (r, c) == (r_new, c_new):
            continue
          if value_map[r_new][c_new]['val'] < val - 1:
            value_map[r_new][c_new] = (val - 1, r, c)
            frontier.append((val - 1, r_new, c_new))
          else:
            frontier.append((value_map[r_new][c_new]['val'], r_new, c_new))
          if grid_map[r_new][c_new] == b'S':
            valid = True

  shortest_path = []
  if valid:
    start = np.asarray(np.where(grid_map == b'S'))
    start_row, start_col = int(start[0]), int(start[1])

    shortest_path = [(start_row, start_col)]
    _, r_prev, c_prev = value_map[start_row, start_col]
    while (r_prev, c_prev) != (-1, -1):
      shortest_path.append((r_prev, c_prev))
      _, r_prev, c_prev = value_map[r_prev, c_prev]


  return valid, value_map, shortest_path

def compute_optimal_policy(grid, value_map):
  optim_policy = np.zeros(grid.shape, dtype=np.str)
  rs, cs = grid.shape
  for r in range(0, rs):
    for c in range(0, cs):
        if grid[r][c] in {b'A', b'G'}:
          optim_policy[r][c] = grid[r][c]
          continue
        min_val = rs*cs
        min_action = 0
        for action in ACTIONS:
          crashed_into_astroid, r_new, c_new = next_pos(grid, r, c, action)
          if crashed_into_astroid:
            continue
          val, _, _ = value_map[r_new][c_new]
          if val < min_val:
            min_val, min_action = val, action
        optim_policy[r][c] = ACTIONS_DESCRIPTION[min_action]
  return optim_policy

def generate_random_map(size=6, p=0.7, optimal_pseudo_rewards=False):
  valid = False
  shortest_path = []
  while not valid:
      p = min(1, p)
      result = np.random.choice([b"-", b"A"], (size, size), p=[p, 1 - p])
      result[-1][0] = b"S"  # set starting point
      result[0][-1] = b"G"  # set goal point
      valid, _, shortest_path = dijkstra(result)
  
  if optimal_pseudo_rewards:
    # remove first and last element
    shortest_path.pop(0)
    shortest_path.pop()
    for idx, (row, col) in enumerate(shortest_path):
      val = idx + 1
      result[row][col] = bytes(str(val).encode('utf-8'))
  
  return result


class IllegalAction(Exception):
    pass

class Environment:
  def __init__(
        self,
        map_name="6x6",
        grid_map=None,
        size=6,
        p=0.7,
        movement_reward=-1,
        asteroid_reward=0,
        goal_reward=10,
        optimal_pseudo_rewards=True,
    ):
      if grid_map is None and map_name is None:
        grid_map = self.generate_random_map(size=size, p=p, optimal_pseudo_rewards=optimal_pseudo_rewards)
      elif grid_map is None:
        grid_map = MAPS[map_name]

      self.grid_map = np.asarray(grid_map, dtype='c')
      self.nrow, self.ncol = numRows, numCols = self.grid_map.shape
      numActions = 8
      numStates = numRows * numCols
      self.nA = numActions
      self.nS = numStates

      path_found, _, self.shortest_path = dijkstra(self.grid_map)
      if not path_found:
        raise IllegalAction(f"grid not valid, no path found!")

      self.min_steps = len(self.shortest_path) - 1
      self.cumulative_reward = 0
      self.cumulative_pseudo_reward = 0
      self.movement_reward = movement_reward
      self.asteroid_reward = asteroid_reward
      self.goal_reward = goal_reward
      self.action_space = np.array(range(0, self.nA))
      self.observation_space = np.array(range(0,self.nS))
      def get_start_state():
        row, col = np.where(self.grid_map == b'S')
        return int(row * self.ncol + col)
      def get_goal_state():
        row, col = np.where(self.grid_map == b'G')
        return int(row * self.ncol + col)
      self.start_state = get_start_state()
      self.goal_state = get_goal_state()
      self.state = self.start_state

  def reset(self):
    self.cumulative_reward = 0
    self.state = self.start_state
    return int(self.state)
  
  def sample_action(self):
    return random.choice(self.action_space)

  def step(self, action):
    if not action in self.action_space:
      raise IllegalAction(f"action {action} is not possible")

    def coords_to_flat_idx(row, col):
      return int(row * self.ncol + col)

    # state to gird pos
    row, col = self.state // self.ncol, self.state % self.ncol
    crashed_into_astroid, r_new, c_new = next_pos(self.grid_map, row, col, action)

    done = False
    reward = self.movement_reward
    pseudo_reward = 0
    letter = self.grid_map[r_new, c_new]
    if crashed_into_astroid:
      reward += self.asteroid_reward
      done = True
    elif bytes(letter) == b'G':
      reward += self.goal_reward
    elif bytes(letter) in [b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9']:
      pseudo_reward += float(letter) / self.min_steps

    self.state = coords_to_flat_idx(r_new, c_new)
    self.cumulative_reward += reward
    self.cumulative_pseudo_reward += pseudo_reward
    total_reward = reward + pseudo_reward

    return int(self.state), total_reward, done
