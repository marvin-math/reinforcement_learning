import numpy as np
from environment import compute_optimal_policy
from environment import dijkstra
from environment import Environment
from environment import MAPS
import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

max_epsilon = 0.9
min_epsilon = 0.0
decay_rate = 0.9
Total_episodes = 1000
max_steps = 100
learning_rate = 0.99
gamma = 0.96



def train(
  map_name="6x6",
  grid_map=None,
  size=6,
  p=0.7,
  movement_reward=-1,
  asteroid_reward=-10,
  goal_reward=10,
  optimal_pseudo_rewards=True,
  print_state=False
):

  def print_pos(row, col):
    outfile = sys.stdout
    disp_map = env.grid_map.tolist()
    disp_map = [[char.decode('utf-8') for char in line] for line in disp_map]
    disp_map[row][col] = 'X'
    outfile.write("\n".join("".join(line) for line in disp_map) + "\n\n")  

  def choose_action(observation):
    action = 0
    if np.random.uniform(0, 1) < epsilon:
      action = env.sample_action()
    else:
      action = np.argmax(Q[observation, :])
    return action

  def learn(observation, observation2, reward, action):
    prediction = Q[observation, action]
    target = reward + gamma * np.max(Q[observation2, :])
    Q[observation, action] = Q[observation, action] + learning_rate * (target - prediction)

  env = Environment(
          map_name=map_name,
          grid_map=grid_map,
          size=size,
          p=p,
          movement_reward=movement_reward,
          asteroid_reward=asteroid_reward,
          goal_reward=goal_reward,
          optimal_pseudo_rewards=optimal_pseudo_rewards)

  Q = np.zeros((env.nS, env.nA))

  epsilons = np.linspace(max_epsilon, min_epsilon, Total_episodes)
  cumulative_rewards = np.array([])
  for episode in range(Total_episodes):
    obs = env.reset()
    r, c = obs // env.ncol, obs % env.ncol
    t = 1
    epsilon = epsilons[episode]
    if print_state:
        print_pos(r, c)
    while t <= max_steps:
      r, c = obs // env.ncol, obs % env.ncol
      action = choose_action(obs)
      if print_state:
        print_pos(r, c)
      obs2, reward, done = env.step(action)
      learn(obs, obs2, reward, action)
      obs = obs2
      if done:
        if obs == env.goal_state:
          print(episode, "WIN")
        else:
          print(episode, "LOSE")
        cumulative_rewards = np.append(cumulative_rewards, env.cumulative_reward / t)
        break
      t += 1

  return cumulative_rewards


def t_test(list_a, list_b):
  return

def print_value_map(gird_map, value_map):
  print("\noptimal value function:\n")
  readable_value_map = ''.join(''.join(('A'.rjust(3) if gird_map[r,c] == b'A' else str(val['val']).rjust(3) for c, val in enumerate(line))) + '\n' for r, line in enumerate(value_map))
  print(readable_value_map)

def print_optimal_policy(optimal_policy):
  print("\noptimal policy:\n")
  readable_policy = ''.join(''.join((str(val).rjust(3) for c, val in enumerate(line))) + '\n' for r, line in enumerate(optimal_policy))
  print(readable_policy)

# Commands to be implemented after running this file
if __name__ == "__main__":
  # Part 2
  rewards = train("6x6")
  rewards_opt_pr= train("6x6_optimal_pseudo_reward")
  rewards_aprx_pr = train("6x6_approximate_pseudo_reward")
  
  # average cumulated reward 
  average_rewards = rewards / (np.arange(len(rewards)) + 1)

  # average cumulated reward (with pseudo rewards)
  average_rewards_pr = rewards_opt_pr / (np.arange(len(rewards_opt_pr)) + 1)

  # average cumulated reward (with approximate pseudo rewards)
  average_rewards_aprx_pr = rewards_aprx_pr / (np.arange(len(rewards_aprx_pr)) + 1)

  fig, ax = plt.subplots()
  #ax.plot(rewards, 'b', label='average normal rewards')
  #ax.plot(rewards_opt_pr, 'r', label='average optimal pseudo rewards')
  #ax.plot(rewards_aprx_pr, 'g', label='average approximate pseudo rewards')

  x = np.arange(len(rewards))
  m,b = np.polyfit(x,rewards,1)
  ax.plot(x,m*x+b, 'b', label = 'no pseudo-rewards')

  x = np.arange(len(rewards_opt_pr))
  m,b = np.polyfit(x,rewards_opt_pr,1)
  plt.plot(x,m*x+b)
  ax.plot(x,m*x+b, 'r', label = 'optimal pseudo-rewards')

  x = np.arange(len(rewards_aprx_pr))
  m,b = np.polyfit(x,rewards_aprx_pr,1)
  ax.plot(x,m*x+b, 'g', label = 'approximate pseudo-rewards')
  plt.legend(loc="upper left")

  ax.set(xlabel='episode index', ylabel='average reward per step')
  ax.grid()
  plt.show()
  
  t=stats.ttest_ind(rewards,rewards_opt_pr)
  t2=stats.ttest_ind(rewards,rewards_aprx_pr)
  t3=stats.ttest_ind(rewards_opt_pr,rewards_aprx_pr)
  print(t, t2, t3)

