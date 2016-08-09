class AgentConfig(object):
  #scale = 10000
  scale = 1000
  scale = 500
  display = False

  #max_step = 5000 * scale
  max_step = 50 * scale
  memory_size = 10 * scale

  batch_size = 64
  random_start = 30
  cnn_format = 'NCHW'
  discount = 0.995
  target_q_update_step = 1 * scale
  learning_rate = 0.005
  learning_rate_minimum = 0.0025
  learning_rate_decay = 0.995
  learning_rate_decay_step = 5 * scale

  ep_end = 0.1
  ep_start = 1.
  ep_end_t = memory_size

  history_length = 4
  train_frequency = 4
  learn_start = 5. * scale

  min_delta = -1
  max_delta = 1

  double_q = False
  dueling = False

  _test_step = 5 * scale
  _save_step = _test_step * 10
  #_test_step = 50 * scale
  #_save_step = _test_step * 100

class EnvironmentConfig(object):
  env_name = 'Breakout-v0'

  screen_width  = 84
  screen_height = 84
  max_reward = 1.
  min_reward = -1.

class DQNConfig(AgentConfig, EnvironmentConfig):
  model = ''
  pass

class M1(DQNConfig):
  backend = 'tf'
  env_type = 'detail'
  action_repeat = 1

class M2(DQNConfig):
  backend = 'tf'
  env_type = 'simple'
  action_repeat = 1

def get_config(FLAGS):
  if FLAGS.model == 'm1':
    config = M1
  elif FLAGS.model == 'm2':
    config = M2

  for k, v in FLAGS.__dict__['__flags'].items():
    if k == 'gpu':
      if v == False:
        config.cnn_format = 'NHWC'
      else:
        config.cnn_format = 'NCHW'

    if hasattr(config, k):
      setattr(config, k, v)

  return config
