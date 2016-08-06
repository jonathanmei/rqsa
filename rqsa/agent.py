import os
import time
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from .base import BaseModel
from .history import History
from .ops import linear, linear_time_dist, conv2d, conv2d_time_dist, lstm
from .replay_memory import ReplayMemory
from utils import get_time, save_pkl, load_pkl

import cProfile

class Agent(BaseModel):
  lstm_out_size = 256
  lstm_state_size = 512
  def __init__(self, config, environment, sess):
    super(Agent, self).__init__(config)
    self.sess = sess
    self.weight_dir = 'weights'

    self.env = environment
    self.history = History(self.config)
    self.memory = ReplayMemory(self.config, self.model_dir)

    self.reset_lstm_states()

    with tf.variable_scope('step'):
      self.step_op = tf.Variable(0, trainable=False, name='step')
      self.step_input = tf.placeholder('int32', None, name='step_input')
      self.step_assign_op = self.step_op.assign(self.step_input)

    self.build_dqn()

  def reset_lstm_states(self):
    self.lstm_state=np.zeros([1, self.lstm_state_size])
    self.lstm_out_prev=np.zeros([1, 1, self.lstm_out_size])

  def train(self):
    start_step = self.step_op.eval()
    start_time = time.time()

    num_game, self.update_count, ep_reward = 0, 0, 0.
    total_reward, self.total_loss, self.total_q = 0., 0., 0.
    max_avg_ep_reward = 0
    ep_rewards, actions = [], []

    screen, reward, action, terminal = self.env.new_random_game()
    #added by jmei: update internal state immediately
    #self.update_lstm_states(action, reward, screen, terminal)
    #end added

    for _ in range(self.history_length):
      self.history.add(screen)

    for self.step in tqdm(range(start_step, self.max_step), ncols=70, initial=start_step):
      if self.step == self.learn_start:
        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        ep_rewards, actions = [], []

      # 1. predict
      action = self.predict_train(self.history.get())
      # 2. act
      screen, reward, terminal = self.env.act(action, is_training=True)
      # 3. observe (LEARNING HAPPENS HERE)
      self.observe(screen, reward, action, terminal)

      if terminal:
        screen, reward, action, terminal = self.env.new_random_game()
        self.reset_lstm_states()
        self.observe(screen,reward,action,terminal)

        num_game += 1
        ep_rewards.append(ep_reward)
        ep_reward = 0.
      else:
        ep_reward += reward

      actions.append(action)
      total_reward += reward

      if self.step >= self.learn_start:
        if self.step % self.test_step == self.test_step - 1:
          avg_reward = total_reward / self.test_step
          avg_loss = self.total_loss / self.update_count
          avg_q = self.total_q / self.update_count

          try:
            max_ep_reward = np.max(ep_rewards)
            min_ep_reward = np.min(ep_rewards)
            avg_ep_reward = np.mean(ep_rewards)
          except:
            max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

          print '\navg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # game: %d' \
              % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_game)

          if max_avg_ep_reward * 0.9 <= avg_ep_reward:
            self.step_assign_op.eval({self.step_input: self.step + 1})
            self.save_model(self.step + 1)

            max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)

          if self.step > 180:
            self.inject_summary({
                'average.reward': avg_reward,
                'average.loss': avg_loss,
                'average.q': avg_q,
                'episode.max reward': max_ep_reward,
                'episode.min reward': min_ep_reward,
                'episode.avg reward': avg_ep_reward,
                'episode.num of game': num_game,
                'episode.rewards': ep_rewards,
                'episode.actions': actions,
                'training.learning_rate': self.learning_rate_op.eval({self.learning_rate_step: self.step}),
              }, self.step)

          num_game = 0
          total_reward = 0.
          self.total_loss = 0.
          self.total_q = 0.
          self.update_count = 0
          ep_reward = 0.
          ep_rewards = []
          actions = []

  def predict(self, lstm_out, test_ep=None):
    ep = test_ep or (self.ep_end +
        max(0., (self.ep_start - self.ep_end)
          * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))

    if random.random() < ep:
      action = random.randrange(self.env.action_size)
    else:
      action = self.e_q_action.eval({self.e_lstm_out: lstm_out})[0,-1]

    return action

  def predict_train(self, s_t, test_ep=None):
    ep = test_ep or (self.ep_end +
        max(0., (self.ep_start - self.ep_end)
          * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))

    if random.random() < ep:
      action = random.randrange(self.env.action_size)
    else:
      action = self.q_action.eval({self.s_t: [s_t], self.init_state: self.lstm_state})[0,-1]
    return action

  def update_lstm_states(self, act_prev, rew_prev, screen, terminal):
    if not terminal:
      self.lstm_out_prev, self.lstm_state = self.sess.run([self.e_lstm_out, self.lstm_state_end],{self.lstm_state_start: self.lstm_state, self.e_s_t: np.expand_dims(np.expand_dims(screen,0),0)})
    else:
      self.reset_lstm_states()

  def observe(self, screen, reward, action, terminal):
    reward = max(self.min_reward, min(self.max_reward, reward))

    self.history.add(screen)
    self.memory.add(screen, reward, action, terminal)
    #self.update_lstm_states(action,reward,screen,terminal)
    if self.step > self.learn_start:
      if self.step % self.train_frequency == 0:
        self.q_learning_mini_batch() #LEARNING HAPPENS HERE

      if self.step % self.target_q_update_step == self.target_q_update_step - 1:
        self.update_target_q_network()
  

  ################################
  def do_cprofile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats('cumtime')
    return profiled_func
  ###################################
  @do_cprofile

  def q_learning_mini_batch(self): #THIS IS SCHOOL, LEARNING HAPPENS HERE
    if self.memory.count < self.history_length:
      return
    else:
      s_t, action, reward, s_t_plus_1, terminal = self.memory.sample()

    t = time.time()
    q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})
    terminal = np.array(terminal) + 0.
    max_q_t_plus_1 = np.max(q_t_plus_1, axis=2)
    target_q_t = (1. - terminal) * self.discount * max_q_t_plus_1 + reward
    _, q_t, loss, summary_str = self.sess.run([self.optim, self.q, self.loss, self.q_summary], {
      self.target_q_t: target_q_t,
      self.action: action,
      self.s_t: s_t,
      self.learning_rate_step: self.step,
      self.init_state: np.zeros([self.batch_size, self.lstm_state_size]),
    })
    self.update_q_eval()

    '''
    _, p_t, loss_p, summary_str_p = self.sess.run([self.optim_p, self.p, self.loss_p, self.p_summary], {
      self.lstm_out_t: lstm_out_t,
      self.action: action,
      self.lstm_out_1_plus_1: lstm_out_t_plus_1,
      self.learning_rate_step: self.step,
    })
    '''

    self.writer.add_summary(summary_str, self.step)
    self.total_loss += loss
    self.total_q += q_t.mean()
    self.update_count += 1

  def build_dqn(self):
    self.w = {}
    self.t_w = {}
    self.e_w = {}

    #initializer = tf.contrib.layers.xavier_initializer()
    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu

    # training network
    with tf.variable_scope('prediction'):
      if self.cnn_format == 'NHWC':
        self.s_t = tf.placeholder('float32',
            [None, self.screen_width, self.screen_height, self.history_length], name='s_t')
      else:
        self.s_t = tf.placeholder('float32',
            [None, self.history_length, self.screen_width, self.screen_height], name='s_t')

      self.action_0=tf.placeholder('int64',
            [None, self.history_length,1], name='action_0')
      action_0_one_hot=tf.squeeze(tf.one_hot(self.action_0, self.env.action_size, 1.0, 0.0, name='action_0_one_hot'), [2])
      self.reward_0=tf.placeholder('float32',
            [None, self.history_length,1], name='reward_0')

      self.l1, self.w['l1_w'], self.w['l1_b'] = conv2d_time_dist(tf.expand_dims(self.s_t, 2),
          32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='l1')
      self.l2, self.w['l2_w'], self.w['l2_b'] = conv2d_time_dist(self.l1,
          64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='l2')
      self.l3, self.w['l3_w'], self.w['l3_b'] = conv2d_time_dist(self.l2,
          64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='l3')
      shape = self.l3.get_shape().as_list()
      self.l3_flat = tf.reshape(self.l3, [-1, shape[1], reduce(lambda x,y:x*y, shape[2:])])

      #self.lstm_in = tf.cast(tf.concat(2, [self.l3_flat,action_0_one_hot,self.reward_0]), tf.float32)
      self.lstm_in=self.l3_flat
      ############################ LSTM requires explicit batch size, so need 2 different batch sizes for train and predict
      ##### or we can do a dumb quick hack and copy the input batch_size times and run it through the training model
      self.init_state = tf.placeholder('float32', [None, self.lstm_state_size])
      self.lstm_out, _, self.w['lstm_w'], self.w['lstm_b'] = lstm(self.lstm_in, self.init_state, self.lstm_out_size, name='lstm')


      self.l4, self.w['l4_w'], self.w['l4_b'] = linear_time_dist(self.lstm_out, 64, activation_fn=activation_fn, name='l4')
      print(self.l4.get_shape())
      self.q, self.w['q_w'], self.w['q_b'] = linear_time_dist(self.l4, self.env.action_size, name='q')

      self.q_action = tf.argmax(self.q, dimension=2)


      q_summary = []
      avg_q = tf.reduce_mean(self.q, [0,1])
      for idx in xrange(self.env.action_size):
        q_summary.append(tf.histogram_summary('q/%s' % idx, avg_q[idx]))
      self.q_summary = tf.merge_summary(q_summary, 'q_summary')


    # eval network
    with tf.variable_scope('eval'):
      if self.cnn_format == 'NHWC':
        self.e_s_t = tf.placeholder('float32',
            [None, self.screen_width, self.screen_height, 1], name='e_s_t')
      else:
        self.e_s_t = tf.placeholder('float32',
            [None, 1, self.screen_width, self.screen_height], name='e_s_t')

      self.e_action_0=tf.placeholder('int64',
            [None, 1,1], name='e_action_0')
      e_action_0_one_hot=tf.squeeze(tf.one_hot(self.e_action_0, self.env.action_size, 1.0, 0.0, name='e_action_0_one_hot'), [2])
      self.e_reward_0=tf.placeholder('float32',
            [None, 1,1], name='e_reward_0')

      self.e_l1, self.e_w['l1_w'], self.e_w['l1_b'] = conv2d_time_dist(tf.expand_dims(self.e_s_t, 2),
          32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='e_l1')
      self.e_l2, self.e_w['l2_w'], self.e_w['l2_b'] = conv2d_time_dist(self.e_l1,
          64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='e_l2')
      self.e_l3, self.e_w['l3_w'], self.e_w['l3_b'] = conv2d_time_dist(self.e_l2,
          64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='e_l3')

      shape = self.e_l3.get_shape().as_list()
      self.e_l3_flat = tf.reshape(self.e_l3, [-1, shape[1], reduce(lambda x,y:x*y, shape[2:])])

      #self.e_lstm_in = tf.concat(2, [self.e_l3_flat,e_action_0_one_hot,self.e_reward_0])
      self.e_lstm_in=self.e_l3_flat
      self.lstm_state_start=tf.placeholder('float32', [None, self.lstm_state_size])
      ############################ LSTM requires explicit batch size, so need 2 models for train(self.batch_size) and predict(1)
      self.e_lstm_out, self.lstm_state_end, self.e_w['lstm_w'], self.e_w['lstm_b'] = lstm(self.e_lstm_in, self.lstm_state_start, self.lstm_out_size, name='e_lstm') #here we use self.lstm_state

      self.e_l4, self.e_w['l4_w'], self.e_w['l4_b'] = linear_time_dist(self.e_lstm_out, 64, activation_fn=activation_fn, name='l4')
      self.e_q, self.e_w['q_w'], self.e_w['q_b'] = linear_time_dist(self.e_l4, self.env.action_size, name='e_q')
      self.e_q_action = tf.argmax(self.e_q, dimension=2)
    with tf.variable_scope('pred_to_eval'):
      self.e_w_input = {}
      self.e_w_assign_op = {}

      for name in self.w.keys():
        self.e_w_input[name] = tf.placeholder('float32', self.e_w[name].get_shape().as_list(), name=name)
        self.e_w_assign_op[name] = self.e_w[name].assign(self.e_w_input[name])

    # target network
    with tf.variable_scope('target'):
      if self.cnn_format == 'NHWC':
        self.target_s_t = tf.placeholder('float32', 
            [None, self.screen_width, self.screen_height, self.history_length], name='target_s_t')
      else:
        self.target_s_t = tf.placeholder('float32', 
            [None, self.history_length, self.screen_width, self.screen_height], name='target_s_t')

      self.target_action_0=tf.placeholder('int64',
            [None, self.history_length,1], name='target_action_0')
      target_action_0_one_hot=tf.squeeze(tf.one_hot(self.target_action_0, self.env.action_size, 1.0, 0.0, name='target_action_0_one_hot'), [2])
      self.target_reward_0=tf.placeholder('float32',
            [None, self.history_length,1], name='target_reward_0')

      self.target_l1, self.t_w['l1_w'], self.t_w['l1_b'] = conv2d_time_dist(tf.expand_dims(self.target_s_t, 2), 
          32, [8, 8], [4, 4], initializer, activation_fn, self.cnn_format, name='target_l1')
      self.target_l2, self.t_w['l2_w'], self.t_w['l2_b'] = conv2d_time_dist(self.target_l1,
          64, [4, 4], [2, 2], initializer, activation_fn, self.cnn_format, name='target_l2')
      self.target_l3, self.t_w['l3_w'], self.t_w['l3_b'] = conv2d_time_dist(self.target_l2,
          64, [3, 3], [1, 1], initializer, activation_fn, self.cnn_format, name='target_l3')
      shape = self.target_l3.get_shape().as_list()
      self.target_l3_flat = tf.reshape(self.target_l3, [-1, shape[1], reduce(lambda x,y:x*y, shape[2:])])

      #self.target_lstm_in = tf.concat(2, [self.target_l3_flat,target_action_0_one_hot,self.target_reward_0])
      self.target_lstm_in=self.target_l3_flat

      target_init_state = tf.zeros([self.config.batch_size, self.lstm_state_size])
      self.target_lstm_out, _, self.t_w['lstm_w'], self.t_w['lstm_b'] = lstm(self.target_lstm_in, target_init_state, self.lstm_out_size, name='target_lstm')
      


      self.target_l4, self.t_w['l4_w'], self.t_w['l4_b'] = \
          linear_time_dist(self.target_lstm_out, 64, activation_fn=activation_fn, name='target_l4')
      self.target_q, self.t_w['q_w'], self.t_w['q_b'] = \
          linear_time_dist(self.target_l4, self.env.action_size, name='target_q')

    with tf.variable_scope('pred_to_target'):
      self.t_w_input = {}
      self.t_w_assign_op = {}

      for name in self.w.keys():
        self.t_w_input[name] = tf.placeholder('float32', self.t_w[name].get_shape().as_list(), name=name)
        self.t_w_assign_op[name] = self.t_w[name].assign(self.t_w_input[name])

    # optimizer
    with tf.variable_scope('optimizer'):
      self.target_q_t = tf.placeholder('float32', [None, self.history_length], name='target_q_t')
      self.action = tf.placeholder('int64', [None, self.history_length], name='action')
      action_one_hot = tf.one_hot(self.action, self.env.action_size, 1.0, 0.0, name='action_one_hot')
      q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices=2, name='q_acted')

      self.delta = self.target_q_t - q_acted
      self.clipped_delta = tf.clip_by_value(self.delta, self.min_delta, self.max_delta, name='clipped_delta')

      self.global_step = tf.Variable(0, trainable=False)

      self.loss = tf.reduce_mean(tf.square(self.clipped_delta), name='loss')
      self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
      self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
          tf.train.exponential_decay(
              self.learning_rate,
              self.learning_rate_step,
              self.learning_rate_decay_step,
              self.learning_rate_decay,
              staircase=True))
      '''
      self.optim = tf.train.RMSPropOptimizer(
          self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)
      '''
      self.optim = tf.train.AdamOptimizer(
          self.learning_rate_op, beta1=0.95, beta2=0.99, epsilon=0.01).minimize(self.loss)
      
      '''
      s_t_plus_1_hat = self.p
      self.delta_p = self.s_t_plus_1 - s_t_plus_1_hat
      self.clipped_delta_p = tf.clip_by_value(self.delta_p, self.min_delta_p, self.max_delta_p, name='clipped_delta')
      self.loss_p = tf.reduce_mean(tf.square(self.clipped_delta_p), name='loss_p')
      self.optim_p = tf.train.RMSPropOptimizer(
          self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss_p)
      '''

    with tf.variable_scope('summary'):
      scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
          'episode.max reward', 'episode.min reward', 'episode.avg reward', 'episode.num of game', 'training.learning_rate']

      self.summary_placeholders = {}
      self.summary_ops = {}

      for tag in scalar_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.scalar_summary("%s-%s/%s" % (self.env_name, self.env_type, tag), self.summary_placeholders[tag])

      histogram_summary_tags = ['episode.rewards', 'episode.actions']

      for tag in histogram_summary_tags:
        self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag.replace(' ', '_'))
        self.summary_ops[tag]  = tf.histogram_summary(tag, self.summary_placeholders[tag])

      self.writer = tf.train.SummaryWriter('./logs/%s' % self.model_dir, self.sess.graph)

    tf.initialize_all_variables().run()

    self._saver = tf.train.Saver(self.w.values() + [self.step_op], max_to_keep=30)

    self.load_model()
    self.update_target_q_network()

  def update_target_q_network(self):
    for name in self.w.keys():
      self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})

  def update_q_eval(self):
    for name in self.w.keys():
      self.e_w_assign_op[name].eval({self.e_w_input[name]: self.w[name].eval()})

  def save_weight_to_pkl(self):
    if not os.path.exists(self.weight_dir):
      os.makedirs(self.weight_dir)

    for name in self.w.keys():
      save_pkl(self.w[name].eval(), os.path.join(self.weight_dir, "%s.pkl" % name))

  def load_weight_from_pkl(self, cpu_mode=False):
    with tf.variable_scope('load_pred_from_pkl'):
      self.w_input = {}
      self.w_assign_op = {}

      for name in self.w.keys():
        self.w_input[name] = tf.placeholder('float32', self.w[name].get_shape().as_list(), name=name)
        self.w_assign_op[name] = self.w[name].assign(self.w_input[name])

    for name in self.w.keys():
      self.w_assign_op[name].eval({self.w_input[name]: load_pkl(os.path.join(self.weight_dir, "%s.pkl" % name))})

    self.update_target_q_network()

  def inject_summary(self, tag_dict, step):
    summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
      self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
    })
    for summary_str in summary_str_lists:
      self.writer.add_summary(summary_str, self.step)

  def play(self, n_step=10000, n_episode=100, test_ep=None, render=False):
    if test_ep == None:
      test_ep = self.ep_end

    test_history = History(self.config)

    if not self.display:
      gym_dir = '/tmp/%s-%s' % (self.env_name, get_time())
      self.env.env.monitor.start(gym_dir)

    best_reward, best_idx = 0, 0
    for idx in xrange(n_episode):
      screen, reward, action, terminal = self.env.new_random_game()
      #added by jmei: update internal state immediately
      self.update_lstm_states(action, reward, screen, terminal)
      #end added
      current_reward = 0

      for _ in range(self.history_length):
        test_history.add(screen)

      for t in tqdm(range(n_step), ncols=70):
        # 1. predict
        action = self.predict(self.lstm_out_prev, test_ep)
        # 2. act
        screen, reward, terminal = self.env.act(action, is_training=False)
        # 3. observe
        test_history.add(screen)
        self.update_lstm_states(action, reward, screen, terminal)

        current_reward += reward
        if terminal:
          break

      if current_reward > best_reward:
        best_reward = current_reward
        best_idx = idx

      print "="*30
      print " [%d] Best reward : %d" % (best_idx, best_reward)
      print "="*30

    if not self.display:
      self.env.env.monitor.close()
      #gym.upload(gym_dir, writeup='https://github.com/devsisters/DQN-tensorflow', api_key='')
