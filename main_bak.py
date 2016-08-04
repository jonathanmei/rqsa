#!/usr/bin/env python
# jmei_ai.py

import sys, os
from random import randrange
from ale_python_interface import ALEInterface
import numpy
import time
import argparse as ap
import rqsa
import cv2

import cProfile

#parse arguments
parser = ap.ArgumentParser(description='jmei\'s AI')
parser.add_argument('-B', '--batch_size', default=32, type=int, required=False, help='Size of mini-batch (default: 32)')
parser.add_argument('-D', '--depth', default=1, type=int, required=False, help='Depth to look ahead if -U is set (default: 1)')
parser.add_argument('-M', '--memory_size', default=10000, required=False, help='Memory size (default: 10000)')
parser.add_argument('-N', '--name_rom', required=True, help='Path to .bin file (Required)')
parser.add_argument('-P', '--predict', action='store_true', required=False, help='Predict using trained model (default: False)')
parser.add_argument('-Q', '--q_val', action='store_true', required=False, help='Use Q-function (default: False)')
parser.add_argument('-R', '--reps', default=500000, type=int, required=False, help='Repetitions of game (default: 500000)')
parser.add_argument('-S', '--steps', default=32, type=int, required=False, help='Steps between training (default: 32)')
parser.add_argument('-T', '--train', action='store_true', required=False, help='Train the network after playing (default: False)')
parser.add_argument('-U', '--unroll', default=8, required=False, help='Length of steps to propagate RNN gradients (default: 8)')
parser.add_argument('-V', '--view', action='store_true', required=False, help='Visualize game window (default: False)')
#parser.add_argument('-W', '--write_data', action='store_true', required=False, help='Write data to file (default: False)')
parser.add_argument('-X', '--xseed', default=88, required=False, help='Random seed (default: 88)')
parser.add_argument('-Y', '--yepsilon', default=0.05, type=float, required=False, help='Epsilon-greedy (default: 0.05)')


parser.add_argument('-r', '--args_rnn_feat_dim', default=128, type=int, required=False, help='rnn feature size (default: 128)')
parser.add_argument('-o', '--args_o_feat_dim', default=128, type=int, required=False, help='cnn output size (default: 128)')
parser.add_argument('-l', '--args_lam', default=0.5, type=float, required=False, help='tradeoff b/n model & q-val (default: 0.5)')
parser.add_argument('-n', '--args_nc', default=1, type=int, required=False, help='num of color channels (default: 1)')
parser.add_argument('-w', '--args_w', default=84, type=int, required=False, help='screen width into cnn (default: 84)')
parser.add_argument('-h', '--args_h', default=84, type=int, required=False, help='screen height into cnn (default: 84)')

pa=parser.parse_args() #pa = parsed args
print(pa)

#start arcade learning environment
ale = ALEInterface()

# Get & Set the desired settings
ale.setInt('random_seed', pa.xseed)
USE_SDL = pa.view #USE_SDL = False #headless! USE_SDL = True # visualize!
if USE_SDL:
  if sys.platform == 'darwin':
    import pygame
    pygame.init()
    ale.setBool('sound', False) # Sound doesn't work on OSX
  elif sys.platform.startswith('linux'):
    ale.setBool('sound', True)
  ale.setBool('display_screen', True)

# Load the ROM file
ale.loadROM(pa.name_rom)
game_name=pa.path_rom.split('/')[-1].split('.')[0]
print(game_name)


# Get the list of actions
actions = ale.getLegalActionSet()
nact=len(actions) #18

#load rqs agent
scr_dims=(pa.args_w, pa.args_h)

rqs=rqsa.RQSAgent(game_name, nact, pa)
if (pa.use_model | pa.q_val):
	rqs.build_model_pieces(0)


ave_len_of_ep=3200
#training params
tt=1


# Play N episodes
tot_timer=time.clock()
tot_reward=0
a=0		#initial action
reward=0 	#initial reward
for rep_num in xrange(pa.reps):
	if pa.train:
		cur_epsilon=1-pa.yepsilon*ep_num/(pa.rep_num-1) #linearly decreasing randomness, from 1 to pa.yepsilon
	else:
		cur_epsilon=pa.yepsilon
	
	screen=ale.getScreenGrayscale()
	o = cv2.resize(cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)/255., scr_dims) #preprocess to 84 x 84 grayscale in [0,1]
        
	if (not pa.train) && (pa.use_model | pa.q_val):
		rqs.update_state(o,a,reward,terminal)

	if not (pa.use_model | pa.q_val): #random actions
		a = actions[randrange(len(actions))]
	else:
		a = rqs.pol(pa.q_val, pa.use_model, pa.depth, cur_epsilon)
	

	# Apply an action and get the resulting reward
	reward = ale.act(a)

	# is terminal state?
	terminal=ale.game_over()

	if pa.train:
		rqs.remember(o,a,reward,terminal)

	total_reward += reward

	if ale.game_over():
		rqs.reset_ep_mem()
		ale.reset_game()
	
	if pa.train:
		print('Training')
		training_timer=time.clock()
		rqs.train_batch(tt, ep, it, NN)
		print('Done training, time: ' + str(time.clock()-training_timer))

print('rewards: ' + str(rewards))
print('Total time: ' + str(time.clock()-tot_timer))
print('All done!')













