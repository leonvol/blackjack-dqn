from DqnAgent import DQN_Agent
from Exploration import EpsilonGreedy
import Dqn

_exploration = EpsilonGreedy(epsilon=0.7, decay=0.01, end_episode=400000)
_episodes = 500000


def compare_nn_linear():
    layers = DQN.generate_layers_linear(5, 5, 10, 3)
    print("Start {}-NN Training".format(layers))
    agent = DQN_Agent(learning_rate=0.01,
                      layers=layers,
                      gamma=0.09,
                      batch_size=256,
                      exploration=_exploration,
                      replay_buffer_cap=10000,
                      ckpt_interval=10000,
                      print_interval=1000,
                      evaluation_steps=1000,
                      name="(10,3)")
    agent.train(episodes=_episodes, collects_per_episode=500)

    layers = Dqn.generate_layers_linear(5, 5, 15, 3)
    print("Start {}-NN Training".format(layers))
    agent = DQN_Agent(learning_rate=0.01,
                      layers=layers,
                      gamma=0.09,
                      batch_size=256,
                      exploration=_exploration,
                      replay_buffer_cap=10000,
                      chkpt_interval=10000,
                      print_interval=1000,
                      evaluation_steps=1000,
                      name="(15,3)")
    agent.train(episodes=_episodes, collects_per_episode=500)

    layers = DQN.generate_layers_linear(5, 5, 15, 3)
    print("Start {}-NN Training".format(layers))
    agent = DQN_Agent(learning_rate=0.01,
                      layers=layers,
                      gamma=0.09,
                      batch_size=256,
                      exploration=_exploration,
                      replay_buffer_cap=10000,
                      ckpt_interval=10000,
                      print_interval=1000,
                      evaluation_steps=1000,
                      name="(15,3)")
    agent.train(episodes=_episodes, collects_per_episode=500)


compare_nn_linear()
