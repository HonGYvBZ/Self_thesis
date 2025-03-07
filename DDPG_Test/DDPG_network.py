import os
import tensorflow as tf
import numpy as np
from tensorflow.initializers import random_uniform


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.5, theta=0.5, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)

        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape)) # input维度为357
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))   # 输出维度为7
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size   # 计算放在数组的第几个
        self.state_memory[index] = state    # 记录当前状态
        self.new_state_memory[index] = state_   # 记录下一个状态
        self.action_memory[index] = action  # 记录当前动作
        self.reward_memory[index] = reward  # 记录奖励
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)   # 积攒了64次经验后才进行抽取
        # 从经验回放区中随机取样
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class Actor(object):
    def __init__(self, lr, n_actions, name, input_dims, sess, fc1_dims,
                 fc2_dims, action_bound, batch_size=64, chkpt_dir='tmp/ddpg'):
        self.lr = lr
        self.n_actions = n_actions  # 指输出的动作 默认为7
        self.name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.sess = sess
        self.action_bound = action_bound
        self.build_network()
        self.params = tf.trainable_variables(scope=self.name)
        self.unnormalized_actor_gradients = tf.gradients(self.mu, self.params, -self.action_gradients)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.actor_gradients, self.params))

    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32, shape=[None, *self.input_dims], name='inputs')
            self.action_gradients = tf.placeholder(tf.float32, shape=[None, self.n_actions], name='gradients')
            f1 = 1. / np.sqrt(self.fc1_dims)
            dense1 = tf.layers.dense(self.input, units=self.fc1_dims, kernel_initializer=random_uniform(-f1, f1),
                                     bias_initializer=random_uniform(-f1, f1))
            batch1 = tf.layers.batch_normalization(dense1)
            layer1_activation = tf.nn.tanh(batch1)
            f2 = 1. / np.sqrt(self.fc2_dims)
            dense2 = tf.layers.dense(layer1_activation, units=self.fc2_dims, kernel_initializer=random_uniform(-f2, f2),
                                     bias_initializer=random_uniform(-f2, f2))
            batch2 = tf.layers.batch_normalization(dense2)
            layer2_activation = tf.nn.tanh(batch2)
            f3 = 1. / np.sqrt(self.fc2_dims)
            dense3 = tf.layers.dense(layer2_activation, units=32, kernel_initializer=random_uniform(-f3, f3),
                                     bias_initializer=random_uniform(-f3, f3))
            batch3 = tf.layers.batch_normalization(dense3)
            layer3_activation = tf.nn.tanh(batch3)
            mu1 = tf.layers.dense(layer3_activation, units=self.n_actions, activation='tanh',
                                  kernel_initializer=random_uniform(-f3, f3), bias_initializer=random_uniform(-f3, f3))
            self.mu = mu1

    def predict(self, inputs):
        return self.sess.run(self.mu, feed_dict={self.input: inputs})   # 输出网络预测的状态（机器人或AP行为）

    def train(self, inputs, gradients):
        self.sess.run(self.optimize, feed_dict={self.input: inputs, self.action_gradients: gradients})


class Critic(object):
    def __init__(self, lr, n_actions, name, input_dims, sess, fc1_dims, fc2_dims,
                 batch_size=64, chkpt_dir='tmp/ddpg'):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.fc1_dims = fc1_dims    # 维度32
        self.fc2_dims = fc2_dims    # 维度32
        self.input_dims = input_dims    # 输入维度357
        self.batch_size = batch_size    # 64
        self.sess = sess
        self.build_network()
        self.params = tf.trainable_variables(scope=self.name)
        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.action_gradients = tf.gradients(self.q, self.actions)

    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32,
                                        shape=[None, *self.input_dims],
                                        name='inputs')
            self.actions = tf.placeholder(tf.float32,
                                          shape=[None, self.n_actions],
                                          name='actions')
            self.q_target = tf.placeholder(tf.float32,
                                           shape=[None, 1],
                                           name='targets')
            f1 = 1. / np.sqrt(self.fc1_dims)
            dense1 = tf.layers.dense(self.input, units=self.fc1_dims,
                                     kernel_initializer=random_uniform(-f1, f1),
                                     bias_initializer=random_uniform(-f1, f1))
            batch1 = tf.layers.batch_normalization(dense1)
            layer1_activation = tf.nn.relu(batch1)
            f2 = 1. / np.sqrt(self.fc2_dims)
            dense2 = tf.layers.dense(layer1_activation, units=self.fc2_dims,
                                     kernel_initializer=random_uniform(-f2, f2),
                                     bias_initializer=random_uniform(-f2, f2))
            batch2 = tf.layers.batch_normalization(dense2)
            action_in = tf.layers.dense(self.actions, units=self.fc2_dims,
                                        activation='relu')
            state_actions = tf.add(batch2, action_in)
            state_actions = tf.nn.relu(state_actions)
            f3 = 0.003
            self.q = tf.layers.dense(state_actions, units=1,
                                     kernel_initializer=random_uniform(-f3, f3),
                                     bias_initializer=random_uniform(-f3, f3),
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01))
            self.loss = tf.losses.mean_squared_error(self.q_target, self.q)

    def predict(self, inputs, actions):
        return self.sess.run(self.q, feed_dict={self.input: inputs, self.actions: actions}) # 预测函数

    def train(self, inputs, actions, q_target):
        return self.sess.run(self.optimize, feed_dict={self.input: inputs, self.actions: actions,
                                                       self.q_target: q_target})    # 进行一次训练

    def get_action_gradients(self, inputs, actions):
        return self.sess.run(self.action_gradients, feed_dict={self.input: inputs, self.actions: actions})


class Agent(object):
    def __init__(self, name_actor, name_critic, name_target_actor, name_target_critic, Lr_A, Lr_C, input_dims, tau,
                 gamma=0.0, n_actions=7, max_size=100000, layer1_size=32, layer2_size=32, batch_size=64,
                 chkpt_dir='tmp/ddpg'):
        self.gamma = gamma      # gamma？常数0？
        self.tau = tau          # tau 常数0.001 平滑数
        self.memory = ReplayBuffer(max_size, input_dims, n_actions) # 设置经验回放缓冲区 100次任务卸载*1000运动=100000 输入单元默认357个
        self.batch_size = batch_size    # 设置batch size为64
        self.sess = tf.Session()    # 使用tf库生成一个会话，这个会话可以执行run

        self.actor = Actor(Lr_A, n_actions, name_actor, input_dims, self.sess,
                           layer1_size, layer2_size, 1, batch_size=64, chkpt_dir=chkpt_dir) # 创建actor 全连接层32维
        self.critic = Critic(Lr_C, n_actions, name_critic, input_dims, self.sess,
                             layer1_size, layer2_size, chkpt_dir=chkpt_dir) # 创建critic
        self.actor_params = self.actor.params
        self.target_actor = Actor(Lr_A, n_actions, name_target_actor, input_dims, self.sess,
                                  layer1_size, layer2_size, 1, batch_size=64, chkpt_dir=chkpt_dir)  # 创建t_actor
        self.target_critic = Critic(Lr_C, n_actions, name_target_critic, input_dims, self.sess,
                                    layer1_size, layer2_size, chkpt_dir=chkpt_dir)  # 创建t_critic
        self.noise = OUActionNoise(mu=np.zeros(n_actions))  # 添加噪音
        self.update_critic = \
            [self.target_critic.params[i].assign(
                tf.multiply(self.critic.params[i], self.tau) \
                + tf.multiply(self.target_critic.params[i], 1.0 - self.tau))
                for i in range(len(self.target_critic.params))] # 定义软更新函数 new_param = tau * current_param + (1 - tau) * target_param
        self.update_actor = \
            [self.target_actor.params[i].assign(
                tf.multiply(self.actor.params[i], self.tau) \
                + tf.multiply(self.target_actor.params[i], 1.0 - self.tau))
                for i in range(len(self.target_actor.params))]
        self.SetParamters = [self.actor.params[i].assign(tf.multiply(self.actor_params[i], 1)) for i in
                             range(len(self.actor.params))]     # 设置参数
        self.sess.run(tf.global_variables_initializer())    # 先跑一次全局初始化
        self.update_network_parameters(first=True)  # 更新网络参数
        self.set_params()   # 设置参数

    def update_network_parameters(self, first=True):
        if first:
            old_tau = self.tau
            self.tau = 0.001    # 设定平滑权重
            self.target_critic.sess.run(self.update_critic) # 进行软更新
            self.target_actor.sess.run(self.update_actor)   # 进行软更新

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)   # 记录四元组到缓存

    def noise(self):
        return self.noise()

    def choose_action(self, state):
        state = state[np.newaxis, :]
        mu = self.actor.predict(state)
        mu_prime = mu  # + self.noise()
        # mu_prime = np.clip((mu + 1) / 2, 0., 1.)
        # print("\n\n")
        # print("mu    ", mu_prime, "  \n\n mu[0]", mu_prime[0])
        # print("--------------------------------------------------------------")
        return mu_prime[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)     # 从经验缓冲区中抽样抽样
        critic_value_ = self.target_critic.predict(new_state, self.target_actor.predict(new_state))
        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma * critic_value_[j] * done[j])
        target = np.reshape(target, (self.batch_size, 1))
        _ = self.critic.train(state, action, target)
        a_outs = self.actor.predict(state)
        grads = self.critic.get_action_gradients(state, a_outs)

        self.actor.train(state, grads[0])
        self.update_network_parameters(first=True)

    def get_param(self):
        return self.actor.params

    def set_params(self):
        self.actor.sess.run(self.SetParamters)


class Federated_Server(object):
    def __init__(self, name_actor, name_critic, input_dims, n_actions=7, layer1_size=32, layer2_size=32):
        self.sess = tf.Session()
        self.actor = Actor(1, n_actions, name_actor, input_dims, self.sess, layer1_size, layer2_size, 1)
        self.actor_params0 = self.actor.params
        self.actor_params1 = self.actor.params
        self.actor_params2 = self.actor.params
        self.actor_params3 = self.actor.params
        self.actor_params4 = self.actor.params
        self.actor_params5 = self.actor.params
        self.actor_params6 = self.actor.params
        self.actor_params7 = self.actor.params
        self.actor_params8 = self.actor.params
        self.actor_params9 = self.actor.params
        self.actor_params10 = self.actor.params
        self.actor_params11 = self.actor.params
        self.actor_params12 = self.actor.params
        self.actor_params13 = self.actor.params
        self.actor_params14 = self.actor.params

        self.ServerFederation = [self.actor.params[i].assign((tf.multiply(self.actor_params0[i], 1) +
                                                              tf.multiply(self.actor_params1[i], 1) +
                                                              tf.multiply(self.actor_params2[i], 1) +
                                                              tf.multiply(self.actor_params3[i], 1) +
                                                              tf.multiply(self.actor_params4[i], 1) +
                                                              tf.multiply(self.actor_params5[i], 1) +
                                                              tf.multiply(self.actor_params6[i], 1) +
                                                              tf.multiply(self.actor_params7[i], 1) +
                                                              tf.multiply(self.actor_params8[i], 1) +
                                                              tf.multiply(self.actor_params9[i], 1) +
                                                              tf.multiply(self.actor_params10[i], 1) +
                                                              tf.multiply(self.actor_params11[i], 1) +
                                                              tf.multiply(self.actor_params12[i], 1) +
                                                              tf.multiply(self.actor_params13[i], 1) +
                                                              tf.multiply(self.actor_params14[i], 1)) / 15) for i in
                                 range(len(self.actor.params))]

        self.sess.run(tf.global_variables_initializer())

    def federation(self):
        self.actor.sess.run(self.ServerFederation)
        return self.actor.params


class Federated_Server_AP(object):
    def __init__(self, name_actor, name_critic, input_dims, n_actions=8, layer1_size=32, layer2_size=32):
        self.sess = tf.Session()
        self.actor = Actor(1, n_actions, name_actor, input_dims, self.sess, layer1_size, layer2_size, 1)
        self.actor_params1 = self.actor.params
        self.actor_params2 = self.actor.params
        self.actor_params3 = self.actor.params
        self.actor_params4 = self.actor.params
        self.ServerFederation_AP = [self.actor.params[i].assign(tf.multiply(self.actor_params1[i], 1) +
                                                                tf.multiply(self.actor_params2[i], 1) +
                                                                tf.multiply(self.actor_params3[i], 1) +
                                                                tf.multiply(self.actor_params4[i], 1) / 4) for i in
                                    range(len(self.actor.params))]
        self.sess.run(tf.global_variables_initializer())

    def federation(self):
        self.actor.sess.run(self.ServerFederation_AP)
        return self.actor.params
