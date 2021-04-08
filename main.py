import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")
env.reset()

def print_observation_space(env):
    print(f"Observation space high: {env.observation_space.high}")
    print(f"Observation space low: {env.observation_space.low}")
    print(f"Number of actions in the action space: {env.action_space.n}")

#Helper function to get true max velocities
def get_max_velocity(env):
    max_velo_cart = 0
    max_velo_pole = 0
    env.reset()
    done = False
    while not done:
        new_state, _, done, _ = env.step(1)
        if (abs(new_state[1]) > max_velo_cart):
            max_velo_cart = abs(new_state[1])
        if abs(new_state[3]) > max_velo_pole:
            max_velo_pole = abs(new_state[3])
        env.render()
    print(f"Max_velo_cart={max_velo_cart}")
    print(f"Max_velo_pole={max_velo_pole}")

def get_discrete_state(state, real_os, disc_os_win_size):
    trimmed_state = np.array([state[2], state[3]])
    discrete_state = (trimmed_state + real_os) / disc_os_win_size
    return tuple(discrete_state.astype(np.int))

def draw_plot(list_in, filename):
    plt.plot(list_in)
    plt.axis([0,EPISODES/LOG_FREQUENCY,0,450])
    plt.xlabel(f"Number of episodes / {LOG_FREQUENCY}")
    plt.ylabel("Reward")
    plt.savefig(filename)

EPISODES = 15000
LOG_FREQUENCY = 100

def train(os_size_0 = 25, os_size_1 = 25, pole_angular_velocity = 3.5, init_value_low = 0, init_value_high = 1, learning_rate = 0.1, discount=0.95, epsilon = 0.1, start_decay = 1, end_decay_at = 0.5):

    discrete_os_size = [os_size_0, os_size_1] #our dimensions
    real_observation_space = np.array([env.observation_space.high[2], pole_angular_velocity]) #disregarding cart data
    discrete_os_win_size = (real_observation_space * 2 / discrete_os_size) #step-size inside our discrete observation space

    q_table = np.random.uniform(low=0, high=1, size =(discrete_os_size + [env.action_space.n]))

    epsilon_decay_by = epsilon / (EPISODES*end_decay_at - start_decay)

    sum = 0
    average_reward_per = []
    for episode in range(EPISODES):
        #Just some logging info
        if episode % LOG_FREQUENCY == 0:
            average_reward_per.append(sum/LOG_FREQUENCY)
            sum = 0
        if episode % (LOG_FREQUENCY*10) == 0:
            print(f"Episode: {episode}")

        #Resetting the environment as well as getting state 0
        discrete_state = get_discrete_state(env.reset(), real_observation_space, discrete_os_win_size)
        done = False

        #One iteration of the environment
        while not done:

            #Using epsilon to introduce exploration
            if np.random.random() > epsilon:
                action = np.argmax(q_table[discrete_state])
            else:
                action = np.random.randint(0,2)

            new_state, reward, done, _ = env.step(action)
            sum += 1
            new_discrete_state = get_discrete_state(new_state, real_observation_space, discrete_os_win_size)

            # Adjusting the values in our Q-table according to the Q-learning formula
            if not done:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action, )]

                new_q = (1-learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
                q_table[discrete_state + (action, )] = new_q

            discrete_state = new_discrete_state

        #Decay epsilon
        if EPISODES*end_decay_at >= episode >= start_decay:
            epsilon -= epsilon_decay_by

    return average_reward_per

draw_plot(train(epsilon=0.9), "plot.png")
