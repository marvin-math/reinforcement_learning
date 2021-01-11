import numpy as np
import heapq
import sys

REGULAR_ACTIONS = [0, 1, 2, 3]
BOOST_ACTIONS = [4, 5, 6, 7]
ACTIONS = REGULAR_ACTIONS + BOOST_ACTIONS

DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1), (4, 0), (0, 4),(-4, 0),  (0, -4)]
BOOST_TO_DIRECTION = {4: (1, 0), 5: (0, 1), 6: (-1, 0), 7: (0, -1)}

ACTIONS_STR = ['d', 'r', 'u', 'l', 'D', 'R', 'U', 'L']

#
# Nomenclature:
# 'S' = starting point
# 'G' = goal point
# 'A' = asteroid
# '-' = empty space
# 'x' = pseudo-reward tile
#
MAP = [
        "AA-A8G",
        "-2A678",
        "-345AA",
        "--A--A",
        "A--A--",
        "S1----",
    ]

grid = np.asarray(MAP, dtype="c")


def action_valid(r, c, action):
  rs, cs = grid.shape
  if action in BOOST_ACTIONS:
    steps = 4
    dr, dc = BOOST_TO_DIRECTION[action]
  else:
    steps = 1
    dr, dc = DIRECTIONS[action]
  r_new, c_new = r, c
  for _ in range(0, steps):
      r_new += dr
      c_new += dc
      if r_new < 0 or r_new >= rs or c_new < 0 or c_new >= cs:
        return False
      if grid[r_new][c_new] == b"A":
        return False
  return True


def print_pos(row, col):
  outfile = sys.stdout
  grid_map = np.full(grid.shape, "0", dtype='c')
  grid_map[row][col] = 'X'
  grid_map = [[char.decode("utf-8") for char in line] for line in grid_map]
  outfile.write("\n".join("".join(line) for line in grid_map) + "\n\n")  

def compute_value_map(grid):
  value_map = np.zeros(grid.shape, dtype=np.int)
  rs, cs = grid.shape

  frontier, discovered = [], set()
  goal = np.where(grid == b'G')
  start_row, start_col = int(goal[0]), int(goal[1])
  value_map[start_row, start_col] = 10

  frontier.append((10, start_row, start_col))

  heapq.heapify(frontier)
  
  # frontier[] tracks the path from the earliest fully discovered tile
  # if frontier[] becomes empty then we have exhausted all valid paths
  while frontier:
    val, r, c = heapq.heappop(frontier)
    if not (r, c) in discovered:
        discovered.add((r, c))
        for action in ACTIONS:
            if action_valid(r, c, action):
              dr, dc = DIRECTIONS[action]
              r_new = r + dr
              c_new = c + dc
              if value_map[r_new][c_new] < val - 1:
                frontier.append((val - 1, r_new, c_new))
                value_map[r_new][c_new] = val - 1
              else:
                frontier.append((value_map[r_new][c_new], r_new, c_new))
                      
  return value_map

def compute_optimal_policy(grid, value_map):
  optim_policy = np.zeros(grid.shape, dtype=np.str)
  rs, cs = grid.shape
  for r in range(0, rs):
    for c in range(0, cs):
      if grid[r][c] == b'A':
        optim_policy[r, c] = 'A'
      else:
        max_val = 0
        optim_action = '-'
        for action in ACTIONS:
          if action_valid(r, c, action):
            dr, dc = DIRECTIONS[action]
            r_new = r + dr
            c_new = c + dc
            if value_map[r_new][c_new] > max_val:
              max_val, optim_action = value_map[r_new][c_new], ACTIONS_STR[action]
        optim_policy[r][c] = optim_action
  return optim_policy



value_map = compute_value_map(grid)
print("\noptimal value function:\n")
readable_value_map = ''.join(''.join(('A'.rjust(3) if grid[r,c] == b'A' else str(val).rjust(3) for c, val in enumerate(line))) + '\n' for r, line in enumerate(value_map))
print(readable_value_map)

optim_policy = compute_optimal_policy(grid, value_map)

print("\noptimal policy:\n")
readable_policy = ''.join(''.join((str(val).rjust(3) for c, val in enumerate(line))) + '\n' for r, line in enumerate(optim_policy))
print(readable_policy)