import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

def conv2d(x,
           output_dim,
           kernel_size,
           stride,
           initializer=tf.contrib.layers.xavier_initializer(),
           activation_fn=tf.nn.relu,
           data_format='NHWC',
           padding='VALID',
           name='conv2d'):
  with tf.variable_scope(name):
    if data_format == 'NCHW':
      stride = [1, 1, stride[0], stride[1]]
      kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[1], output_dim]
    elif data_format == 'NHWC':
      stride = [1, stride[0], stride[1], 1]
      kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

    w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
    conv = tf.nn.conv2d(x, w, stride, padding, data_format=data_format)

    b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.01))
    out = tf.nn.bias_add(conv, b, data_format)

  if activation_fn != None:
    out = activation_fn(out)

  return out, w, b
####################################################
def conv2d_time_dist(x,
           output_dim,
           kernel_size,
           stride,
           initializer=tf.contrib.layers.xavier_initializer(),
           activation_fn=tf.nn.relu,
           data_format='NHWC',
           padding='VALID',
           name='conv2d_time_dist'):
  shape = x.get_shape().as_list()
  steps = shape[1]
  out_tensors=[0]*steps
  with tf.variable_scope(name):
    if data_format == 'NCHW': #then input will be NCcHW, C is time steps, c is channels (e.g. 1 for gray, 3 for rgb)
      stride = [1, 1, stride[0], stride[1]]
      kernel_shape = [kernel_size[0], kernel_size[1], shape[2], output_dim]
    elif data_format == 'NHWC': #then input will be NcHWC
      stride = [1, stride[0], stride[1], 1]
      kernel_shape = [kernel_size[0], kernel_size[1], shape[1], output_dim]

    w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)

    b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.01))

    for i in xrange(steps):
      out_tensors[i] = tf.nn.conv2d(x[:,i,:,:,:], w, stride, padding, data_format=data_format)
      tf.get_variable_scope().reuse_variables()
    out = tf.transpose(tf.pack(out_tensors), (1,0,2,3,4))
    out = tf.nn.bias_add(out, b, data_format)


  if activation_fn != None:
    out = activation_fn(out)

  return out, w, b
#######################################################
def linear(input_, output_size, stddev=0.02, bias_start=0.01, activation_fn=None, name='linear'):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(name):
    w = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
        tf.random_normal_initializer(stddev=stddev))
    b = tf.get_variable('bias', [output_size],
        initializer=tf.constant_initializer(bias_start))

    out = tf.nn.bias_add(tf.matmul(input_, w), b)

    if activation_fn != None:
      out = activation_fn(out)

    return out, w, b

def linear_time_dist(input_, output_size, stddev=0.02, bias_start=0.01, activation_fn=None, name='linear_time_dist'):
  shape = input_.get_shape().as_list()
  steps = shape[1]
  out_tensors=[0]*steps
  # time steps is shape[1]
  with tf.variable_scope(name):
    w = tf.get_variable('Matrix', [shape[2], output_size], tf.float32,
        tf.random_normal_initializer(stddev=stddev))
    b = tf.get_variable('bias', [output_size],
        initializer=tf.constant_initializer(bias_start))

    for i in xrange(steps):
      out_tensors[i] = tf.matmul(input_[:,i,:], w)
      tf.get_variable_scope().reuse_variables()
    out = tf.transpose(tf.pack(out_tensors), (1,0,2))
    out = tf.nn.bias_add(out, b)

    if activation_fn != None:
      out = activation_fn(out)

    return out, w, b

def lstm(inp, init_state, output_size, name='lstm'):
  shape = inp.get_shape().as_list()
  steps = shape[1]
  state = init_state
  cell = tf.nn.rnn_cell.BasicLSTMCell(output_size)
  out_tensors = [0]*steps
  with tf.variable_scope(name) as vs:
    
    for i in range(steps):
      out_tensors[i], state = cell(inp[:,i,:], state)
      tf.get_variable_scope().reuse_variables()

    lstm_vars = [v for v in tf.all_variables() if v.name.startswith(vs.name)]
    w=lstm_vars[0]
    b=lstm_vars[1]
    out = tf.transpose(tf.pack(out_tensors), (1,0,2))
    return out, state, w, b
