import keras.layers as kl
import keras.models as km
import keras.backend as kb
import numpy, h5py, os, math, csv
from random import randrange
import time
import cPickle as pickle
class RQSAgent:

	#constants
	#nc=1 #num channels
	#w=210 #width of screen
	#h=160 #height of screen
	def __init__(self, game_name, nact, pa, dscnt=0.95):

		self.fpath=os.path.join(os.path.dirname(__file__),game_name+'_data')
		
		self.model={}
		self.data={}
		self.memory=rm.ReplayMemory(pa)
		self.nact=nact

		self.nc=pa.args_nc
		self.w=pa.args_w
		self.h=pa.args_h

		self.T=pa.args_unroll
		self.lam=pa.args_lam
		self.rnn_out_dim=pa.args_rnn_feat_dim
		self.cnn_o_out_dim=pa.args_o_feat_dim

		self.cnn_P_out_dim=self.rnn_out_dim+1 #+1 for reward prediction :)
		self.cnn_Q_out_dim=nact
		self.cnn_a_out_dim=nact

		self.dscnt=dscnt
		self.batch_size=pa.batch_size
		self.wtname='rqsa_'+game_name+'_rnn'+str(self.rnn_out_dim)+'_o'+str(self.cnn_o_out_dim)+'_wt_'
		
		self.mem=rm.ReplayMemory(pa)
		
		self.reset_state()
		# affects:
		#  self.suff
		#  self.model['RNN']

		if not os.path.exists(self.fpath):
			os.makedirs(self.fpath)

	##### data and file utils #####
	def reset_state(self):
		self.suff=numpy.zeros([1,self.rnn_out_dim])
		if 'RNN' in self.model:
			self.model['RNN'].reset_states()


	def embed_a(self, a): #creates one-hot vector
		N=len(a)
		label_acts=[[0]*self.nact]*N
		for i,v in enumerate(a):
			label_acts[i][v]=1
		return label_acts


	def remember(o,a,r,t):
		self.memory.add(o,a,r,t)
# model pieces, modularized because training and evaluation are slightly different
# 	train: use batches from replay_mem, in which we look at sequences of length self.T using keras TimeDistributed layer wrapper
# 	eval: step through one time step

	def build_model_pieces(self,train_eval):
		#cnns, used to estimate Q and P from "sufficient" statistic output by RNN
		in_Q=kl.Input(shape=(self.rnn_out_dim,))
		x=kl.Dense(2*self.nact, activation='relu')(in_Q)
		x=kl.Dropout(0.1)(x)
		out_Q=kl.Dense(self.cnn_Q_out_dim)(x)
		self.model['CNN_Q']=km.Model(input=in_Q,output=out_Q)
		print('cnn_q ok')

		in_P=kl.Input(shape=(self.rnn_out_dim+self.nact,))
		x=kl.Dense(self.rnn_out_dim+self.nact/2, activation='relu')(in_P)
		x=kl.Dropout(0.1)(x)
		out_P=kl.Dense(self.cnn_P_out_dim)(x)
		self.model['CNN_P']=km.Model(input=in_P,output=out_P)
		print('cnn_p ok')

		#rnn used to keep track of "sufficient" statistic for observation sequence history
		n_steps=[1,self.T][train_eval] #single RNN cell
		
		if train_eval:
			self.model['RNN']=km.Sequential()
			self.model['RNN'].add(kl.recurrent.GRU(self.rnn_out_dim, dropout_W=0.1, dropout_U=0.1, return_sequences=True, stateful=False, input_shape=(n_steps,self.cnn_o_out_dim+self.cnn_a_out_dim+1))) # + 1 for reward
		else:
			self.model['RNN']=km.Sequential()
			self.model['RNN'].add(kl.recurrent.GRU(self.rnn_out_dim, dropout_W=0.1, dropout_U=0.1, return_sequences=True, stateful=True, batch_input_shape=(1,n_steps,self.cnn_o_out_dim+self.cnn_a_out_dim+1))) #first index is explicit batch size of 1
		print('rnn ok')

		#cnn, used to embed (obs, act) into RNN input space
		self.model['CNN_o']=km.Sequential()
		self.model['CNN_o'].add(kl.Convolution2D(32,8,8,border_mode='valid', subsample=(4,4), activation='relu', input_shape=(self.nc,self.w,self.h)))
		self.model['CNN_o'].add(kl.Dropout(0.1))
		self.model['CNN_o'].add(kl.Convolution2D(32,4,4,border_mode='valid', subsample=(4,4), activation='relu'))
		self.model['CNN_o'].add(kl.Dropout(0.1))
		self.model['CNN_o'].add(kl.Convolution2D(32,4,4,border_mode='valid', subsample=(2,2), activation='relu'))
		self.model['CNN_o'].add(kl.Flatten())
		self.model['CNN_o'].add(kl.Dense(self.cnn_o_out_dim))
		print('cnn_o ok')
		# pass actions straight through:
		self.model['CNN_a']=km.Sequential()
		self.model['CNN_a'].add(kl.Activation('linear',input_shape=(self.cnn_a_out_dim,)))
		print('cnn_a ok')
		self.load_weights()
		
	#returns a model architecture for training
	def build_model_train(self): #lam is tradeoff b/n Q-fn and PSR in loss function for training

		input_a=kl.Input(shape=(self.T,self.nact))
		output_a=kl.wrappers.TimeDistributed(self.model['CNN_a'])(input_a)

		input_o=kl.Input(shape=(self.T,self.nc,self.w,self.h))
		output_o=kl.wrappers.TimeDistributed(self.model['CNN_o'])(input_o)

		input_r = kl.Input(shape=(self.T,1))
		output_r = kl.Activation('linear')(input_r)

		input_rnn=kl.merge([output_o, output_a, output_r],mode='concat',concat_axis=2)
		output_rnn=self.model['RNN'](input_rnn)

		out_Q=kl.wrappers.TimeDistributed(self.model['CNN_Q'])(output_rnn)
		out_Q_reshaped=kl.Reshape((self.T,self.nact))(out_Q)
		print('out_Q ok')
		input_a_next=kl.Input(shape=(self.T,self.nact)) #one-hot action, is actually full a not a_til
		out_a_next=kl.Reshape((self.T,self.nact))(input_a_next)
		print('input_a_next ok')
		out_rew=kl.merge([out_Q_reshaped,out_a_next], mode=lambda x : kb.batch_dot(x[0], x[1], axes=2), output_shape=lambda x: (None, self.T, 1))
		print('out_rew ok')

		model_train_Q=km.Model(input=[input_o, input_a, input_r, input_a_next], output=out_rew)

		input_suff1=kl.Input(shape=(self.T-1, self.rnn_out_dim))
		input_a1=kl.Input(shape=(self.T-1, self.nact))
		input_P=kl.merge([input_suff1, input_a1],mode='concat', concat_axis=2)
		print('input_P ok')
		out_P=kl.wrappers.TimeDistributed(self.model['CNN_P'])(input_P)
		print('out_P ok')		
		model_train_P=km.Model(input=[input_suff1, input_a1], output=out_P)
		

		return model_train_Q, model_train_P

		#rmsprop, adam work; adadelta, SGD not on this prob

	def train(self, tt, ep, it, NN):

		#tt = num of test examples to hold out (not yet cross-validation)
		#ep = num of epochs per iteration to train (also how stale target network is AND how often to save weights)
		#it = num of total iterations of outer loop to train
		#NN = num of random example runs of length self.T to generate from saved data

		import re,fnmatch
		old_wts=fnmatch.filter(os.listdir(self.fpath), self.wtname+'*')
		vers=[ int([i for i in re.split(self.wtname+'|CNN_o|CNN_a|CNN_Q|CNN_P|RNN|.h5',j) if i][-1]) for j in old_wts ]
		cur_ver=0
		if vers:
			cur_ver=numpy.amax(vers)
		print('vers: '+str(cur_ver))
		NN_names=['CNN_o','CNN_a','CNN_Q','CNN_P','RNN']		
		for nm in NN_names:
			if os.path.isfile(os.path.join(self.fpath,self.wtname+nm+str(cur_ver)+'.h5')):
				self.model[nm].load_weights(os.path.join(self.fpath,self.wtname+nm+str(cur_ver)+'.h5'))

		
		self.build_model_pieces(1) #1 denotes we are training, so the rnn will have multiple time steps and be unrolled
		self.save_data(NN)

		mod_Q, mod_P=self.build_model_train()
		# output is batch_size x o_til x cnn_o_out_dim
		mod_P.compile(optimizer='rmsprop', loss='mse')
		mod_Q.compile(optimizer='rmsprop', loss='mse')
		
		for i in range(it):
			cnn_o_in=kl.Input(shape=(self.T,self.nc,self.w,self.h))
			cnn_o_out=kl.TimeDistributed(self.model['CNN_o'])(cnn_o_in)
			CNN_o_time_dist=km.Model(input=cnn_o_in,output=cnn_o_out)
			cnn_a_in=kl.Input(shape=(self.T,self.nact))
			cnn_a_out=kl.TimeDistributed(self.model['CNN_a'])(cnn_a_in)
			CNN_a_time_dist=km.Model(input=cnn_a_in,output=cnn_a_out)
			Q_in=kl.Input(shape=(self.T,self.rnn_out_dim))
			Q_out=kl.TimeDistributed(self.model['CNN_Q'])(Q_in)
			Q_time_dist=km.Model(input=Q_in, output=Q_out)

			for o_data,a_data,r_data in self.data_gen(NN):

				#build_data returns sequences of T+1 consecutive states
				o=o_data[:,0:self.T,...]
				a=a_data[:,0:self.T,...]
				r=r_data[:,0:self.T,...]
				o_next=o_data[:,1:self.T+1,...]
				a_next=a_data[:,1:self.T+1,...]
				r_next=r_data[:,1:self.T+1,...]

				
				o_til_next=CNN_o_time_dist.predict(o_next)

				a_til_next=CNN_a_time_dist.predict(a_next)

				stale_rnn_out_next = self.model['RNN'].predict(numpy.concatenate([o_til_next,a_til_next,r_next],axis=2), batch_size=self.batch_size)

				
				stale_Q_next = Q_time_dist.predict(stale_rnn_out_next)
				stale_Q_target = r_next + numpy.expand_dims(self.dscnt*numpy.amax(stale_Q_next,axis=(2,)),axis=2)

				mod_Q.fit([o,a,r,a_next], stale_Q_target, nb_epoch=ep, batch_size=self.batch_size)

			cnn_o_in=kl.Input(shape=(self.T,self.nc,self.w,self.h))
			cnn_o_out=kl.TimeDistributed(self.model['CNN_o'])(cnn_o_in)
			CNN_o_time_dist=km.Model(input=cnn_o_in,output=cnn_o_out)
			cnn_a_in=kl.Input(shape=(self.T,self.nact))
			cnn_a_out=kl.TimeDistributed(self.model['CNN_a'])(cnn_a_in)
			CNN_a_time_dist=km.Model(input=cnn_a_in,output=cnn_a_out)
			Q_in=kl.Input(shape=(self.T,self.rnn_out_dim))
			Q_out=kl.TimeDistributed(self.model['CNN_Q'])(Q_in)
			Q_time_dist=km.Model(input=Q_in, output=Q_out)
			
			for o_data,a_data,r_data in self.data_gen(NN):
				#build_data returns sequences of T+1 consecutive states
				o_next=o_data[:,1:self.T+1,...]
				a_next=a_data[:,1:self.T+1,...]
				r_next=r_data[:,1:self.T+1,...]

				cnn_o_in=kl.Input(shape=(self.T,self.nc,self.w,self.h))
				cnn_o_out=kl.TimeDistributed(self.model['CNN_o'])(cnn_o_in)
				CNN_o_time_dist=km.Model(input=cnn_o_in,output=cnn_o_out)
				o_til_next=CNN_o_time_dist.predict(o_next)

				cnn_a_in=kl.Input(shape=(self.T,self.nact))
				cnn_a_out=kl.TimeDistributed(self.model['CNN_a'])(cnn_a_in)
				CNN_a_time_dist=km.Model(input=cnn_a_in,output=cnn_a_out)
				a_til_next=CNN_a_time_dist.predict(a_next)

				stale_rnn_out_next = self.model['RNN'].predict(numpy.concatenate([o_til_next,a_til_next,r_next],axis=2), batch_size=self.batch_size)
				suff1 = stale_rnn_out_next[:,:-1,...]
				suff2 = stale_rnn_out_next[:,1:,...]

				mod_P.fit([suff1,a_next[:,:-1,...]], numpy.append(r_next[:,:-1,...], suff2, axis=2), nb_epoch=ep, batch_size=self.batch_size)

			NN_names=['CNN_o','CNN_a','CNN_Q','CNN_P','RNN']		
			for nm in NN_names:
				self.model[nm].save_weights(os.path.join(self.fpath,self.wtname+nm+str(cur_ver)),overwrite=True)
			cur_ver = cur_ver+1

		#next section is not needed in current version, here it's just to check performance and debug
		'''
		test_labels_estimated=self.model_val.predict(self.build_state(self.data['test_state_rew_pairs'][:,0]), batch_size=self.batch_size)

		test_err=self.model_val.evaluate(self.build_state(self.data['test_state_rew_pairs'][:,0]),self.data['test_state_rew_pairs'][:,2]+self.dscnt* numpy.amax( self.model_val.predict(self.build_state(self.data['test_state_rew_pairs'][:,1]),batch_size=self.batch_size), axis=1 ),batch_size=self.batch_size)
		'''
		#print performance
		'''
		print('\ntest error: ')
		print(test_err)
		print('\ntrue, est:\r')
		print(numpy.append(self.data['test_state_rew_pairs'][:,0],+self.dscnt* numpy.amax( self.model_val.predict(self.build_state(self.data['test_state_rew_pairs'][:,1]),batch_size=self.batch_size), axis=1 )[0:10],test_labels_estimated[0:10],axis=1))
		'''


	def load_weights(self): #returns trained model from stored weights
		import re,fnmatch
		old_wts=fnmatch.filter(os.listdir(self.fpath), self.wtname+'*')
		vers=[ int([i for i in re.split(self.wtname+'|CNN_o|CNN_a|CNN_Q|CNN_P|RNN|.h5',j) if i][-1]) for j in old_wts ]
		cur_ver=0
		if vers:
			cur_ver=numpy.amax(vers)
		print('vers: '+str(cur_ver))
		NN_names=['CNN_o','CNN_a','CNN_Q','CNN_P','RNN']		
		for nm in NN_names:
			if os.path.isfile(os.path.join(self.fpath,self.wtname+nm+str(cur_ver)+'.h5')):
				self.model[nm].load_weights(os.path.join(self.fpath,self.wtname+nm+str(cur_ver)+'.h5'))

	##### compute policy using networks! #####
	def pol_Q(self, suff):
		return self.model['CNN_Q'].predict(suff) # vector of Q values

	def step_P(self, suff, a):
		return self.model['CNN_P'].predict(numpy.concatenate((suff,self.embed_a([a])),axis=1))

	def pol(self, o, q_val, use_model, steps, epsilon):
		if not numpy.random.binomial(1, epsilon): #pick greedy action
			if (not use_model) | (steps == 0): #q_val only
				return numpy.argmax(self.pol_Q(self.suff)) #greeeeed is good
			else:
				q1=self.pol_Q(self.suff)
				q2=numpy.zeros_like(q1)
				for a in range(self.nact):
					r_suff_next = self.step_P(self.suff,a)

					r=r_suff_next[:,0]
					suff_next=r_suff_next[:,1:]
					q2[:,a]=r+self.dscnt*numpy.amax(self.pol_Q(suff_next))
				return numpy.argmax( (1-self.lam)*(q1)+self.lam*(q2) ) #TODO: tree search, can be parallel
		else: #pick randomly
			return randrange(self.nact)

	def update_state(self, o,a,rew):
		o_til=self.model['CNN_o'].predict(numpy.expand_dims(numpy.transpose(o,axes=(2,0,1)),axis=0))
		self.suff=numpy.squeeze(self.model['RNN'].predict( numpy.expand_dims(numpy.concatenate((o_til,self.embed_a([a]),[[rew]]),axis=1),axis=1)),axis=1)












