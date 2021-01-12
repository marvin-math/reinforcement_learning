# reinforcement learning agent
A reinforcement learning task. The agent can navigate a 6x6 grid. Some tiles contain Asteroids (which terminate the episode) and the agent can activate a boost and move across 4 tiles. Every step has a cost of -1, finding the goal gives a reward of 10. 
The grid looks as follows:  

![6x6grid](https://user-images.githubusercontent.com/59829442/104290294-78aa2980-54ba-11eb-9fc3-3cad410b4c74.PNG)  

The optimal_vf_p file visualizes the optimal value function and the optimal policy  

To improve training effects, pseudo rewards have been implemented (see: Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations: Theory
and application to reward shaping. In ICML (Vol. 99, pp. 278-287))  

The program compares the performance of agents with  
a) no pseudo rewards  
b) approximate pseudo rewards  
c) optimal pseudo rewards  

The subsequent t-tests test whether the performance of these different agents significantly deviates from one another.  

Note that there are several variables that can be played with, such as: gamma, learning rate, max_steps, total_episodes, decay_rate, min_epsilon, max_epsilon
## Installation
Download the environment, main and optimal_vf_p file into the same folder. Run the optimal_vf_p file to have the optimal policy and optimal value function visualized. Run the main file to see the different agents perform.
