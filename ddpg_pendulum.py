import gym.spaces
import numpy as np
import tensorflow as tf
import random
from collections import deque
import time
import os

# start time
start = time.time()

##################################################
################# Initialization #################
##################################################

# make directories for saving data
save_dir = 'saved_data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    os.makedirs(save_dir + '/trained_params')
    os.makedirs(save_dir + '/tensorboard_training')
    os.makedirs(save_dir + '/tensorboard_evaluation')

# create env
env = gym.make('PendulumDisturbance-v0')
env = env.unwrapped

# set seed
env.seed(0)
np.random.seed(0)
tf.set_random_seed(0)

# observation/action dimensions
observation_dim = np.prod(np.array(env.observation_space.shape))
action_dim      = np.prod(np.array(env.action_space.shape))

# observation history dimensions
history_time_span       = 1  # consider 1s history observation/action
history_steps           = int(history_time_span/env.dt)
observation_history_dim = (observation_dim + action_dim)*history_steps + observation_dim

# hyperparameters
h_actor           = 64  # hidden layer size for the actor
h_critic          = 64  # hidden layer size for the critic
lr_actor          = 1e-3  # learning rate for the actor
lr_critic         = 1e-3  # learning rate for the critic
lr_decay          = 1  # learning rate decay per episode
l2_reg_actor      = 1e-6  # L2 regularization factor for the actor
l2_reg_critic     = 1e-6  # L2 regularization factor for the critic
dropout_actor     = 0  # dropout rate for actor (0 = no dropout)
dropout_critic    = 0  # dropout rate for critic (0 = no dropout)
gamma             = 0.99  # reward discount factor
tau               = 1e-2  # soft target update rate
memory_capacity   = int(1e5)  # capacity of experience replay memory
batch_size        = 1024  # size of batch from experience replay memory for updates

# training/evaluation parameters
num_eps_train     = 2000  # number of episodes for training
num_eps_eval      = 100  # number of episodes for evaluation
num_ep_steps      = 200  # default max number of steps per episode (unless env has a lower hardcoded limit)
training_interval = 1  # number of steps to run the policy (and collect experience) before updating network weights
ON_TRAIN          = False  # training or evaluation
RENDER            = False  # render or not

# exploration noise
initial_noise_scale = 0.1  # scale of the exploration noise process (1.0 is the range of each action dimension)
noise_decay         = 0.99  # decay rate (per episode) of the scale of the exploration noise process
noise_mu            = 0.0  # mu parameter for the exploration noise process: dXt = theta*(mu-Xt)*dt + sigma*dWt
noise_theta         = 0.15  # theta parameter for the exploration noise process: dXt = theta*(mu-Xt)*dt + sigma*dWt
noise_sigma         = 0.2  # sigma parameter for the exploration noise process: dXt = theta*(mu-Xt	)*dt + sigma*dWt

##################################################
################ Memory Settings #################
##################################################

memory = deque(maxlen=memory_capacity)  # used for O(1) popleft() operation

def add_to_memory(experience):
    memory.append(experience)

def sample_from_memory(batch_size):
    return random.sample(memory, batch_size)

##################################################
############## Tensorflow Settings ###############
##################################################

tf.reset_default_graph()

# placeholders
observation_history_ph      = tf.placeholder(dtype=tf.float32, shape=[None, observation_history_dim])
action_ph                   = tf.placeholder(dtype=tf.float32, shape=[None, action_dim])
reward_ph                   = tf.placeholder(dtype=tf.float32, shape=[None])
next_observation_history_ph = tf.placeholder(dtype=tf.float32, shape=[None, observation_history_dim])
not_terminal_flag_ph        = tf.placeholder(dtype=tf.float32, shape=[None])  # indicators (go into target computation)
training_flag_ph            = tf.placeholder(dtype=tf.bool,    shape=())      # for dropout

# episode counter
ep_counter = tf.Variable(0.0, trainable=False, name='ep_counter')
ep_inc_op  = ep_counter.assign_add(1)

# will use this to initialize both the actor network its target network with same structure
def build_actor_network(observation_history, trainable, reuse):
    h1      = tf.layers.dense(observation_history, h_actor, activation=tf.nn.relu, trainable=trainable, name='actor_hidden_1', reuse=reuse)
    h1_drop = tf.layers.dropout(h1, rate=dropout_actor, training=trainable & training_flag_ph)

    h2      = tf.layers.dense(h1_drop, h_actor, activation=tf.nn.relu, trainable=trainable, name='actor_hidden_2', reuse=reuse)
    h2_drop = tf.layers.dropout(h2, rate=dropout_actor, training=trainable & training_flag_ph)

    h3      = tf.layers.dense(h2_drop, h_actor, activation=tf.nn.relu, trainable=trainable, name='actor_hidden_3', reuse=reuse)
    h3_drop = tf.layers.dropout(h3, rate=dropout_actor, training=trainable & training_flag_ph)

    action_unscaled = tf.layers.dense(h3_drop, action_dim, trainable=trainable, name='actor_output', reuse=reuse)
    action = env.action_space.low + tf.nn.sigmoid(action_unscaled) * (env.action_space.high - env.action_space.low)  # bound the actions to the valid range

    return action

# will use this to initialize both the critic network its target network with same structure
def build_critic_network(observation_history, action, trainable, reuse):
    critic_input = tf.concat([observation_history, action], axis=1)

    h1      = tf.layers.dense(critic_input, h_critic, activation=tf.nn.relu, trainable=trainable, name='critic_hidden_1', reuse=reuse)
    h1_drop = tf.layers.dropout(h1, rate=dropout_critic, training=trainable & training_flag_ph)

    h2      = tf.layers.dense(h1_drop, h_critic, activation=tf.nn.relu, trainable=trainable, name='critic_hidden_2', reuse=reuse)
    h2_drop = tf.layers.dropout(h2, rate=dropout_critic, training=trainable & training_flag_ph)

    h3      = tf.layers.dense(h2_drop, h_critic, activation=tf.nn.relu, trainable=trainable, name='critic_hidden_3', reuse=reuse)
    h3_drop = tf.layers.dropout(h3, rate=dropout_critic, training=trainable & training_flag_ph)

    q_values = tf.layers.dense(h3_drop, 1, trainable=trainable, name='critic_output', reuse=reuse)

    return q_values

# actor network
with tf.variable_scope('actor_eval'):
    # policy's outputted action for each observation_history_ph (for generating actions and training the critic)
    actions = build_actor_network(observation_history_ph, trainable=True, reuse=False)

# target actor network
with tf.variable_scope('actor_target', reuse=False):
    # target policy's outputted action for each next_observation_history_ph (for training the critic)
    # use stop_gradient to treat the output values as constant targets when doing backprop
    target_next_actions = tf.stop_gradient(build_actor_network(next_observation_history_ph, trainable=False, reuse=False))

# critic network
with tf.variable_scope('critic_eval') as scope:
    # critic applied to observation_history_ph and a given action (for training critic)
    q_values_of_given_actions = build_critic_network(observation_history_ph, action_ph, trainable=True, reuse=False)
    # critic applied to observation_history_ph and the current policy's outputted actions for observation_history_ph (for training actor via deterministic policy gradient)
    q_values_of_suggested_actions = build_critic_network(observation_history_ph, actions, trainable=True, reuse=True)

# target critic network
with tf.variable_scope('critic_target', reuse=False):
    # target critic applied to target actor's outputted actions for next_observation_history_ph (for training critic)
    target_next_q_values = tf.stop_gradient(build_critic_network(next_observation_history_ph, target_next_actions, trainable=False, reuse=False))

# isolate vars for each network
actor_vars         = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_eval')
target_actor_vars  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,    scope='actor_target')
critic_vars        = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_eval')
target_critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,    scope='critic_target')

# update values for targets towards current actor and critic
update_target_ops = []
for i, target_actor_var in enumerate(target_actor_vars):
    update_target_actor_op = target_actor_var.assign(tau * actor_vars[i] + (1 - tau) * target_actor_var)
    update_target_ops.append(update_target_actor_op)

for i, target_critic_var in enumerate(target_critic_vars):
    update_target_critic_op = target_critic_var.assign(tau * critic_vars[i] + (1 - tau) * target_critic_var)
    update_target_ops.append(update_target_critic_op)

update_targets_op = tf.group(*update_target_ops, name='update_targets')

# one step TD targets y_i for (s,a) from experience replay
# = r_i + gamma*Q_slow(s',mu_slow(s')) if s' is not terminal
# = r_i if s' terminal
target_q_values = tf.expand_dims(reward_ph, 1) + tf.expand_dims(not_terminal_flag_ph, 1) * gamma * target_next_q_values

# one-step TD errors
td_errors = target_q_values - q_values_of_given_actions

# critic loss function (mean-square value error with regularization)
critic_loss = tf.reduce_mean(tf.square(td_errors))
for var in critic_vars:
    if not 'bias' in var.name:
        critic_loss += l2_reg_critic * 0.5 * tf.nn.l2_loss(var)

# critic optimizer
critic_train_op = tf.train.AdamOptimizer(lr_critic * lr_decay ** ep_counter).minimize(critic_loss)

# actor loss function (mean Q-values under current policy with regularization)
actor_loss = -1 * tf.reduce_mean(q_values_of_suggested_actions)
for var in actor_vars:
    if not 'bias' in var.name:
        actor_loss += l2_reg_actor * 0.5 * tf.nn.l2_loss(var)

# actor optimizer
# the gradient of the mean Q-values w.r.t. actor params is the deterministic policy gradient (keeping critic params fixed)
actor_train_op = tf.train.AdamOptimizer(lr_actor * lr_decay ** ep_counter).minimize(actor_loss, var_list=actor_vars)

# tensorflow session
sess = tf.Session()

# tensorboard for training
if ON_TRAIN:
    # tensorflow variables
    tf_ep_reward               = tf.Variable(0.0, 'ep_reward')
    tf_ep_actor_loss_running   = tf.Variable(0.0, 'ep_actor_loss_running')
    tf_ep_critic_loss_running  = tf.Variable(0.0, 'ep_critic_loss_running')
    tf_ep_actor_loss_training  = tf.Variable(0.0, 'ep_actor_loss_training')
    tf_ep_critic_loss_training = tf.Variable(0.0, 'ep_critic_loss_training')

    tf.summary.scalar('ep_reward', tf_ep_reward)
    tf.summary.scalar('ep_actor_loss_running', tf_ep_actor_loss_running)
    tf.summary.scalar('ep_critic_loss_running', tf_ep_critic_loss_running)
    tf.summary.scalar('ep_actor_loss_training', tf_ep_actor_loss_training)
    tf.summary.scalar('ep_critic_loss_training', tf_ep_critic_loss_training)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(save_dir + '/tensorboard_training/', sess.graph)

    # run in terminal:
    # tensorboard --logdir Dropbox/UTS/Research\ Progress/Stage\ 8\ Reinforcement\ Learning/Code/PycharmProjects/model_free_rl_v5/saved_data/tensorboard_training/

# tensorboard for evaluation
else:
    # tensorflow variables
    tf_ep_reward              = tf.Variable(0.0, 'ep_reward')
    tf_ep_actor_loss_running  = tf.Variable(0.0, 'ep_actor_loss_running')
    tf_ep_critic_loss_running = tf.Variable(0.0, 'ep_critic_loss_running')

    tf.summary.scalar('ep_reward', tf_ep_reward)
    tf.summary.scalar('ep_actor_loss_running', tf_ep_actor_loss_running)
    tf.summary.scalar('ep_critic_loss_running', tf_ep_critic_loss_running)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(save_dir + '/tensorboard_evaluation/', sess.graph)

    # run in terminal:
    # tensorboard --logdir Dropbox/UTS/Research\ Progress/Stage\ 8\ Reinforcement\ Learning/Code/PycharmProjects/model_free_rl_v5/saved_data/tensorboard_evaluation/

# initialize all tensorflow variables
sess.run(tf.global_variables_initializer())

# make saver
saver = tf.train.Saver()

##################################################
#################### Training ####################
##################################################

# train policy
def train_policy(num_eps):

    # initialize variables
    next_observation         = np.zeros(observation_dim)
    next_observation_history = np.zeros(observation_history_dim)
    reward                   = 0
    done                     = False

    # total steps of all episodes
    total_steps = 0

    for i in range(num_eps):
        # number of episodes
        ep = i + 1

        # reset total reward in each episode
        ep_reward = 0

        # reset total batch actor/critic loss in each episode
        total_batch_actor_loss  = 0
        total_batch_critic_loss = 0

        # reset number of training step in each episode
        ep_train_steps = 0

        # reset running data in each episode
        ep_observation_histories      = []
        ep_actions                    = []
        ep_rewards                    = []
        ep_next_observation_histories = []
        ep_not_terminal_flags         = []

        # initial observation
        observation = env.reset()

        # initial observation history
        observation_history = observation

        # initialize exploration noise process
        noise_process = np.zeros(action_dim)
        noise_scale   = (initial_noise_scale * noise_decay ** ep) * (env.action_space.high - env.action_space.low)

        # disturbance parameters (randomly change in each episode)
        A     = 2 + np.random.random()*1  # amplitude [2,3]
        T     = 2 + np.random.random()*2  # period [2,4]
        omega = 2*np.pi/T  # angular frequency
        phi   = np.random.random()*np.pi  # phase [0,pi]

        for j in range(num_ep_steps):
            # number of steps in each episode
            ep_step = j + 1

            # time in an episode
            ep_t = (ep_step - 1)*env.dt

            # render
            if RENDER and ep % 10 == 0:
                env.render()

            # disturbance (randomly change in each episode)
            disturbance = A*np.sin(omega*ep_t + phi)

            ##################################################
            ################# Random actions #################
            ##################################################

            # when history observation space is not full
            if ep_step <= history_steps:
                # randomly choose action
                action = np.array([np.random.random()*env.max_torque*2 - env.max_torque])

                # take step
                next_observation, reward, done, info = env.step(action, disturbance)

                # add current observation/action to observation history
                next_observation_history = np.concatenate((observation_history, action, next_observation))

            ##################################################
            ################# Start training #################
            ##################################################

            # when history observation space is full
            if ep_step > history_steps:
                # choose action based on deterministic policy
                action, = sess.run(actions, feed_dict={observation_history_ph: observation_history[None], training_flag_ph: False})

                # add temporally-correlated exploration noise to action (using an Ornstein-Uhlenbeck process)
                noise_process = noise_theta * (noise_mu - noise_process) + noise_sigma * np.random.randn(action_dim)
                action += noise_scale * noise_process

                # take step
                next_observation, reward, done, info = env.step(action, disturbance)

                # update observation history
                index = list(range(0, int(observation_dim + action_dim)))
                next_observation_history = np.delete(observation_history, index)
                next_observation_history = np.concatenate((next_observation_history, action, next_observation))

                # decide if next observation is a terminal state
                not_terminal_flag = 0.0 if done else 1.0

                # add to memory
                add_to_memory((observation_history, action, reward, next_observation_history, not_terminal_flag))

                # update network weights to fit a batch of experience
                if total_steps % training_interval == 0 and len(memory) >= batch_size:
                    # update number of training step in each episode
                    ep_train_steps += 1

                    # grab N (s,a,r,s') tuples from replay memory
                    batch = sample_from_memory(batch_size)

                    # update the critic and actor params using mean-square value error and deterministic policy gradient
                    _, _, batch_actor_loss, batch_critic_loss = sess.run([actor_train_op, critic_train_op, actor_loss, critic_loss],
                                                                         feed_dict={observation_history_ph:      np.asarray([elem[0] for elem in batch]),
                                                                                    action_ph:                   np.asarray([elem[1] for elem in batch]),
                                                                                    reward_ph:                   np.asarray([elem[2] for elem in batch]),
                                                                                    next_observation_history_ph: np.asarray([elem[3] for elem in batch]),
                                                                                    not_terminal_flag_ph:        np.asarray([elem[4] for elem in batch]),
                                                                                    training_flag_ph:            True})

                    # update target actor and critic towards current actor and critic
                    _ = sess.run(update_targets_op)

                    # update total batch actor/critic loss in each episode
                    total_batch_actor_loss  += batch_actor_loss
                    total_batch_critic_loss += batch_critic_loss

                # collect running data in each episode
                ep_observation_histories.append(observation_history)
                ep_actions.append(action)
                ep_rewards.append(reward)
                ep_next_observation_histories.append(next_observation_history)
                ep_not_terminal_flags.append(not_terminal_flag)

            ##################################################
            ###################### End #######################
            ##################################################

            # update total reward in each episode
            ep_reward += reward

            # update observation
            observation = next_observation

            # update observation history
            observation_history = next_observation_history

            # update total steps
            total_steps += 1

            if done or ep_step == num_ep_steps:
                # update episode counter
                _ = sess.run(ep_inc_op)

                # turn the running data of each episode into an array
                ep_observation_histories      = np.array(ep_observation_histories)
                ep_actions                    = np.array(ep_actions)
                ep_rewards                    = np.array(ep_rewards)
                ep_next_observation_histories = np.array(ep_next_observation_histories)
                ep_not_terminal_flags         = np.array(ep_not_terminal_flags)

                # running actor/critic loss of each episode
                ep_actor_loss_running, ep_critic_loss_running = sess.run([actor_loss, critic_loss],
                                                                         feed_dict={observation_history_ph:      ep_observation_histories,
                                                                                    action_ph:                   ep_actions,
                                                                                    reward_ph:                   ep_rewards,
                                                                                    next_observation_history_ph: ep_next_observation_histories,
                                                                                    not_terminal_flag_ph:        ep_not_terminal_flags,
                                                                                    training_flag_ph:            False})

                # training actor/critic loss of each episode
                if ep_train_steps == 0:
                    ep_actor_loss_training  = 0
                    ep_critic_loss_training = 0
                else:
                    ep_actor_loss_training  = total_batch_actor_loss/ep_train_steps
                    ep_critic_loss_training = total_batch_critic_loss/ep_train_steps

                # assign tensorflow variables
                assign_reward = tf_ep_reward.assign(ep_reward)
                sess.run(assign_reward)

                assign_actor_loss_running = tf_ep_actor_loss_running.assign(ep_actor_loss_running)
                sess.run(assign_actor_loss_running)

                assign_critic_loss_running = tf_ep_critic_loss_running.assign(ep_critic_loss_running)
                sess.run(assign_critic_loss_running)

                assign_actor_loss_training = tf_ep_actor_loss_training.assign(ep_actor_loss_training)
                sess.run(assign_actor_loss_training)

                assign_critic_loss_training = tf_ep_critic_loss_training.assign(ep_critic_loss_training)
                sess.run(assign_critic_loss_training)

                # tensorboard
                result = sess.run(merged)
                writer.add_summary(result, ep)

                print('Episode: %5i | Steps: %4i | Reward: %9.3f | Actor loss: %9.3f | Critic loss: %9.3f | Noise scale: %6.3f' % (
                    ep, ep_step, ep_reward, ep_actor_loss_running, ep_critic_loss_running, noise_scale))
                break

    # save trained variables
    saver.save(sess, save_dir + '/trained_params/ddpg_pendulum.ckpt')

##################################################
################### Evaluation ###################
##################################################

# evaluate policy
def evaluate_policy(num_eps):

    # restore trained variables
    saver.restore(sess, save_dir + '/trained_params/ddpg_pendulum.ckpt')

    # initialize variables
    next_observation         = np.zeros(observation_dim)
    next_observation_history = np.zeros(observation_history_dim)
    reward                   = 0
    done                     = False

    for i in range(num_eps):
        # number of episodes
        ep = i + 1

        # reset total reward in each episode
        ep_reward = 0

        # reset running data in each episode
        ep_observation_histories      = []
        ep_actions                    = []
        ep_rewards                    = []
        ep_next_observation_histories = []
        ep_not_terminal_flags         = []

        # initial observation
        observation = env.reset()

        # initial observation history
        observation_history = observation

        # disturbance parameters (randomly change in each episode)
        A     = 2 + np.random.random()*1  # amplitude [2,3]
        T     = 2 + np.random.random()*2  # period [2,4]
        omega = 2*np.pi/T  # angular frequency
        phi   = np.random.random()*np.pi  # phase [0,pi]

        for j in range(num_ep_steps):
            # number of steps in each episode
            ep_step = j + 1

            # time in an episode
            ep_t = (ep_step - 1)*env.dt

            # render
            env.render()

            # disturbance (randomly change in each episode)
            disturbance = A*np.sin(omega*ep_t + phi)

            ##################################################
            ################# Random actions #################
            ##################################################

            # when history observation space is not full
            if ep_step <= history_steps:
                # randomly choose action
                action = np.array([np.random.random()*env.max_torque*2 - env.max_torque])

                # take step
                next_observation, reward, done, info = env.step(action, disturbance)

                # add current observation/action to observation history
                next_observation_history = np.concatenate((observation_history, action, next_observation))

            ##################################################
            ################# Start training #################
            ##################################################

            # when history observation space is full
            if ep_step > history_steps:
                # choose action based on deterministic policy
                action, = sess.run(actions, feed_dict={observation_history_ph: observation_history[None], training_flag_ph: False})

                # take step
                next_observation, reward, done, info = env.step(action, disturbance)

                # update observation history
                index = list(range(0, int(observation_dim + action_dim)))
                next_observation_history = np.delete(observation_history, index)
                next_observation_history = np.concatenate((next_observation_history, action, next_observation))

                # decide if next observation is a terminal state
                not_terminal_flag = 0.0 if done else 1.0

                # collect running data in each episode
                ep_observation_histories.append(observation_history)
                ep_actions.append(action)
                ep_rewards.append(reward)
                ep_next_observation_histories.append(next_observation_history)
                ep_not_terminal_flags.append(not_terminal_flag)

            ##################################################
            ###################### End #######################
            ##################################################

            # update total reward in each episode
            ep_reward += reward

            # update observation
            observation = next_observation

            # update observation history
            observation_history = next_observation_history

            if done or ep_step == num_ep_steps:
                # turn the running data of each episode into an array
                ep_observation_histories      = np.array(ep_observation_histories)
                ep_actions                    = np.array(ep_actions)
                ep_rewards                    = np.array(ep_rewards)
                ep_next_observation_histories = np.array(ep_next_observation_histories)
                ep_not_terminal_flags         = np.array(ep_not_terminal_flags)

                # running actor/critic loss of each episode
                ep_actor_loss_running, ep_critic_loss_running = sess.run([actor_loss, critic_loss],
                                                                         feed_dict={observation_history_ph:      ep_observation_histories,
                                                                                    action_ph:                   ep_actions,
                                                                                    reward_ph:                   ep_rewards,
                                                                                    next_observation_history_ph: ep_next_observation_histories,
                                                                                    not_terminal_flag_ph:        ep_not_terminal_flags,
                                                                                    training_flag_ph:            False})

                # assign tensorflow variables
                assign_reward = tf_ep_reward.assign(ep_reward)
                sess.run(assign_reward)

                assign_actor_loss_running = tf_ep_actor_loss_running.assign(ep_actor_loss_running)
                sess.run(assign_actor_loss_running)

                assign_critic_loss_running = tf_ep_critic_loss_running.assign(ep_critic_loss_running)
                sess.run(assign_critic_loss_running)

                # tensorboard
                result = sess.run(merged)
                writer.add_summary(result, ep)

                print('Episode: %5i | Steps: %4i | Reward: %9.3f | Actor loss: %9.3f | Critic loss: %9.3f' % (
                    ep, ep_step, ep_reward, ep_actor_loss_running, ep_critic_loss_running))
                break

##################################################
###################### Main ######################
##################################################

if ON_TRAIN:
    print("\n##################################################")
    print("Training policy")
    print("##################################################\n")

    train_policy(num_eps_train)
else:
    print("\n##################################################")
    print("Evaluating trained policy")
    print("##################################################\n")

    evaluate_policy(num_eps_eval)

##################################################
##################### Timer ######################
##################################################

# end time
end = time.time()

# time used
elapsed = end - start

h = elapsed // 3600
m = (elapsed - h*3600) // 60
s = elapsed - h*3600 - m*60

print("Time used: %2i h - %2i m - %2i s" % (h, m, s))
