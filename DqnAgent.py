import datetime
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.pyplot import figure
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from BlackjackEnv import BlackjackEnv
from Dqn import Dqn
from Exploration import Exploration
from ReplayBuffer import ReplayBuffer


def _create_output_mask(state):
    """:returns mask for valid actions"""
    return np.array([
        True, True, True,
        bool(state[2] and state[3]),
        bool(state[3] and 9 <= state[0] <= 11)
    ])


class DqnAgent(object):
    def __init__(self,
                 exploration: Exploration,
                 layers,
                 learning_rate,
                 replay_buffer_cap,
                 gamma,
                 batch_size,
                 chkpt_interval,
                 print_interval,
                 evaluation_steps,
                 update_freq=4,
                 name=None):

        self.n_actions = BlackjackEnv().action_space
        self.exploration = exploration
        self.gamma = gamma

        self.name = name if name is not None else str(layers)
        self.dir = "models/" + self.name + "/"

        try:
            self.model = tf.keras.models.load_model(self.dir)
        except:
            self.model = Dqn(layers)

        self.chkpt_interval = chkpt_interval
        self.print_interval = print_interval
        self.evaluation_steps = evaluation_steps
        self.train_writer = None
        self.test_writer = None

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=self.optimizer, run_eagerly=True)
        self.replay_buffer = ReplayBuffer(replay_buffer_cap)

        self.step_counter = 0
        self.update_freq = update_freq
        self.batch_size = batch_size

    def loss(self, x, y):
        """calculates loss"""
        return tf.keras.losses.logcosh(y, x)

    def generate(self, n_data=1000, n_parallel=10):
        """
        generates @param n_data data points with @param n_parallel semi parallel workers
        method pushes state-action-reward pairs to replay buffer
        """
        envs = [BlackjackEnv() for _ in range(n_parallel)]
        states = np.array([env.reset()[0] for env in envs])
        data_count = 0

        # semi-parallelize environment actions and predictions by stacking states to predict with dqn
        while True:
            pred_mask = np.array([self.exploration.should_predict(self.step_counter) for _ in range(n_parallel)])
            actions = np.array([0] * n_parallel)
            pred_states = states[pred_mask]

            if np.any(pred_mask == True):
                action_masks = np.array([_create_output_mask(state) for state in pred_states])
                pred_actions = tf.argmax(tf.where(action_masks, self.model(np.array(pred_states)), -16), axis=1)
                actions[pred_mask] = pred_actions

            rand_states = states[np.logical_not(pred_mask)]
            rand_actions = [self.exploration.choose_random(self.n_actions, _create_output_mask(state)) for state in
                            rand_states]
            actions[np.logical_not(pred_mask)] = rand_actions

            # iterate over envs and push pairs to replay buffer
            for i, env in enumerate(envs):
                data_count += 1
                next_state, reward, done = env.step(actions[i])

                self.replay_buffer.push([
                    states[i],
                    actions[i],
                    reward,
                    next_state,
                    done
                ])

                states[i] = next_state

                if data_count == n_data:
                    return

                # reset environment if it is finished
                if done:
                    states[i] = env.reset()[0]

    def train(self, episodes, collects_per_episode):
        """trains the model for @param episodes, collects collects_per_episode per episode"""
        start_time = datetime.datetime.now()
        start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
        output_template = "{} Epoch {}, Average Test Reward: {}"

        # instantiate writers for average evaluation rewards and losses
        self.train_writer = tf.summary.create_file_writer(self.dir + "logs/gradient_tape/" + start_time_str + "/train")
        self.test_writer = tf.summary.create_file_writer(self.dir + "logs/gradient_tape/" + start_time_str + "/test")

        self.generate(n_data=1000, n_parallel=20)
        t_last = time.time()

        for episode in range(episodes):
            self.step_counter += 1

            if self.step_counter % 2 == 0:
                self.generate(n_data=collects_per_episode, n_parallel=int(collects_per_episode / 3))

            if self.step_counter % self.update_freq == 0:
                loss = self.optimize()
                with self.test_writer.as_default():
                    tf.summary.scalar("loss", loss, step=self.step_counter)

            if episode % self.print_interval == 0:
                print("Epoch training took {} secs".format(time.time() - t_last))

                avg_reward = self.evaluate(self.evaluation_steps)
                with self.train_writer.as_default():
                    tf.summary.scalar("accuracy", avg_reward, step=self.step_counter)
                print(output_template.format(datetime.datetime.now(), episode, avg_reward))

                t_last = time.time()

            if episode % self.chkpt_interval == 0:
                self.model.save(self.dir + "checkpoints/{}/".format(episode))

        self.model.save(self.dir + "final-{}".format(episodes))
        info_file = open(self.dir + "info-{}.txt".format(self.name), "wb")
        info_file.write("Training took {}".format(datetime.datetime.now() - start_time))

    def optimize(self):
        """
        optimizes the model based on the bellman equation
        :returns loss
        """
        minibatch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, finished = zip(*minibatch)
        states, actions, rewards, next_states, finished = np.array(states), np.array(actions), np.array(
            rewards), np.array(next_states), np.array(finished)

        with tf.GradientTape() as tape:
            # gathering q values for previous states and actions
            predicted_Q = tf.gather(self.model(states), indices=actions, batch_dims=1, axis=1)
            next_Q = np.zeros(self.batch_size)

            # check if there are non-finished states
            if np.any(finished == False):
                # retrieve non-finished valid states and their masks
                valid_states = next_states[np.logical_not(finished)]
                valid_masks = np.array([_create_output_mask(state) for state in valid_states])

                valid_predictions = tf.where(valid_masks, self.model(valid_states), y=-16)
                predictions_max = tf.reduce_max(valid_predictions, axis=1)
                next_Q[np.logical_not(finished)] = predictions_max

            next_Q = rewards + next_Q * self.gamma
            _loss = self.loss(next_Q, predicted_Q)

        # update gradients
        grads = tape.gradient(_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return _loss

    def evaluate(self, n_episodes=100):
        """
        evaluates model by playing n_episodes games in parallel
        :returns average reward of all games
        """
        envs = np.array([BlackjackEnv() for _ in range(n_episodes)])
        states = np.array([env.reset()[0] for env in envs])

        rewards = np.array([-1] * n_episodes)
        active_mask = np.array([True] * n_episodes)

        step_count = 0
        while np.count_nonzero(active_mask) > 0:
            # break after games take more than 5 steps to complete to avoid endless loop
            if step_count > 5: break

            step_count += 1
            actions = np.empty(shape=[n_episodes])

            # mask states to retrieve active states
            pred_states = states[active_mask]
            action_masks = [_create_output_mask(state) for state in pred_states]
            # choose highest rated action
            pred_actions = np.argmax(tf.where(action_masks, self.model(pred_states), -16), axis=1)
            actions[active_mask] = pred_actions

            # iterate over all envs, retrieve new state or reward if not already finished
            for i, env in enumerate(envs):
                if active_mask[i]:
                    ret = env.step(actions[i])
                    states[i], rewards[i], done = ret
                    if done:
                        active_mask[i] = False

        return np.average(rewards)
