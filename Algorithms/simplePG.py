import numpy as np
import time

#from chainer import cuda

#import cupy as cp

#backend
#be = "gpu"
#device = 0


be = "cpu"

class SimplePG(object):
	
	
	# constructor
	def __init__(self, num_actions, input_size, hidden_layer_size, learning_rate,gamma,decay_rate,greedy_e_epsilon,random_seed):
		# store hyper-params
		self._A = num_actions
		self._D = input_size
		self._H = hidden_layer_size
		self._learning_rate = learning_rate
		self._decay_rate = decay_rate
		self._gamma = gamma
		
		# some temp variables
		self._xs,self._hs,self._dlogps,self._drs = [],[],[],[]

		# some hitl temp variables
		self.h_actions, self.h_rewards = [], []
		self.a_probs = []


		# variables governing exploration
		self._exploration = True # should be set to false when evaluating
		self._explore_eps = greedy_e_epsilon
		
		#create model
		self.init_model(random_seed)
		
	
	def init_model(self,random_seed):
		# create model
		#with cp.cuda.Device(0):
		self._model = {}
		np.random.seed(random_seed)
	   
		# weights from input to hidden layer   
		self._model['W1'] = np.random.randn(self._D,self._H) / np.sqrt(self._D) # "Xavier" initialization
	   
		# weights from hidden to output (action) layer
		self._model['W2'] = np.random.randn(self._H,self._A) / np.sqrt(self._H)
			
				
		self._grad_buffer = { k : np.zeros_like(v) for k,v in self._model.items() } # update buffers that add up gradients over a batch
		self._rmsprop_cache = { k : np.zeros_like(v) for k,v in self._model.items() } # rmsprop memory

	def save_agent(self):
		pass
	
	# softmax function
	def softmax(self,x):
		probs = np.exp(x - np.max(x, axis=1, keepdims=True))
		probs /= np.sum(probs, axis=1, keepdims=True)
		return probs
		
	  
	def discount_rewards(self,r):
		""" take 1D float array of rewards and compute discounted reward """
		discounted_r = np.zeros_like(r)
		running_add = 0
		for t in reversed(range(0, r.size)):
			running_add = running_add * self._gamma + r[t]
			discounted_r[t] = float(running_add)
    
		return discounted_r
	
	# feed input (x) to network and get result
	def policy_forward(self,x):
		if(len(x.shape)==1):
			x = x[np.newaxis,...]

		h = x.dot(self._model['W1'])
		
		if np.isnan(np.sum(self._model['W1'])):
			print("W1 sum is nan")
		
		if np.isnan(np.sum(self._model['W2'])):
			print("W2 sum is nan")
		
		if np.isnan(np.sum(h)):
			print("nan")
			
			h[np.isnan(h)] = np.random.random_sample()
			h[np.isinf(h)] = np.random.random_sample()
			

		if np.isnan(np.sum(h)):
			print("Still nan!")
		
		
		h[h<0] = 0 # ReLU nonlinearity
		logp = h.dot(self._model['W2'])

		p = self.softmax(logp)
  
		return p, h # return probability of taking actions, and hidden state
		
	
	def policy_backward(self, eph, epdlogp):
		""" backward pass. (eph is array of intermediate episode hidden states) 
			epdlogp is the """
		dW2 = eph.T.dot(epdlogp)  
		dh = epdlogp.dot(self._model['W2'].T)
		dh[eph <= 0] = 0 # backpro prelu
  
		t = time.time()
  
		# if(be == "gpu"):
		#   self._dh_gpu = cuda.to_gpu(dh, device=0)
		#   self._epx_gpu = cuda.to_gpu(self._epx.T, device=0)
		#   self._dW1 = cuda.to_cpu(self._epx_gpu.dot(self._dh_gpu) )
		# else:
		
		self._dW1 = self._epx.T.dot(dh) 
		#print(f"dw1: {self._dW1} \n")
    

		#print((time.time()-t0)*1000, ' ms, @final bprop')

		return {'W1':self._dW1, 'W2':dW2}
	
	def set_explore_epsilon(self,e):
		self._explore_eps = e
	
	# input: current state/observation
	# output: action index
	def process_step(self, x, exploring):

		# feed input (x) through network and get output: action probability distribution and hidden layer
		aprob, h = self.policy_forward(x)
		
		#print(aprob)
		
		# if exploring
		if exploring == True:
			
			# greedy-e exploration
			rand_e = np.random.uniform()
			#print(rand_e)
			if rand_e < self._explore_eps:
				# set all actions to be equal probability
				aprob[0] = [ 1.0/len(aprob[0]) for i in range(len(aprob[0]))]
				#print("!")
		
		
		if np.isnan(np.sum(aprob)):
			#print(aprob)
			aprob[0] = [ 1.0/len(aprob[0]) for i in range(len(aprob[0]))]
			#print(aprob)
			#input()
		
		#probabilities are added
		#print(aprob, "aprob normal")

		aprob_cum = np.cumsum(aprob)
		#print(aprob_cum, "aprobc \n")
		u = np.random.uniform()
		#print(u, "u\n")
		a = np.where(u <= aprob_cum)[0][0]	
		#print(a, "\n")

		# record various intermediates (needed later for backprop)
		t = time.time()
		self._xs.append(x) # observation
		self._hs.append(h)

		#Probabilities of actions
		self.a_probs.append(aprob)

		#softmax loss gradient
		dlogsoftmax = aprob.copy()
		dlogsoftmax[0,a] -= 1 #-discounted reward WRONG
		self._dlogps.append(dlogsoftmax)

		
		t  = time.time()

		return a
		
	# after process_step, this function needs to be called to set the reward
	def save_rewards(self,reward,state=None, action=None, next_state=None, done=None):
		
		# store the reward in the list of rewards
		self._drs.append(reward)
		
	# reset to be used when evaluating
	def reset(self):
		self._xs,self._hs,self._dlogps,self._drs = [],[],[],[] # reset 
		self._grad_buffer = { k : np.zeros_like(v) for k,v in self._model.items() } # update buffers that add up gradients over a batch
		self._rmsprop_cache = { k : np.zeros_like(v) for k,v in self._model.items() } # rmsprop memory
		


	# this function should be called when an episode (i.e., a game) has finished
	def finish_episode(self, ep_return = None, human_demonstration=False):
		# stack together all inputs, hidden states, action gradients, and rewards for this episode
		
		# this needs to be stored to be used by policy_backward
		# self._xs is a list of vectors of size input dim and the 
		# number of vectors is equal to the number of time steps in the episode
		self._epx = np.vstack(self._xs)
		#print(f"epx: {self._epx}, shape: {np.shape(self._epx)}")
		#for i in range(0,len(self._hs)):
		#	print(self._hs[i])
		
		# len(self._hs) = # time steps
		# stores hidden state activations
		#print(self._hs)
		eph = np.vstack(self._hs)
		# print(f"eph: {eph} shape: {np.shape(eph)}")
		# print(f"hs: {self._hs} shape: {np.shape(self._hs)}")
		# self._dlogps stores a history of the softmax probabilities over actions selected by the agent
		epdlogp = np.vstack(self._dlogps)

		# self._a_probs stores a history of the probabilities that add to one over actions selected by the agent
		robot_actions = np.vstack(self.a_probs)
		#print(f"epdlogp: {epdlogp}, shape: {np.shape(epdlogp)}")
		
		# self._drs is the history of rewards from the episode
		epr = np.vstack(self._drs)
		#print(f"epr: {epr}")

		if human_demonstration:
			human_actions = np.vstack(self.h_actions)

		
		self._xs,self._hs,self._dlogps,self._drs, self.h_actions, self.a_probs = [],[],[],[],[],[] # reset array memory

		#compute the discounted reward backwards through time
		discounted_epr = (self.discount_rewards(epr))
		#print(f"discounted_epr: {discounted_epr}")

		#Mean of the discounted rewards
		discounted_epr_mean = np.mean(discounted_epr)
		#print(f"discounted_epr_mean: {discounted_epr_mean}")
		
		# standardize the rewards to be unit normal (helps control the gradient estimator variance)
		discounted_epr_diff = np.subtract(discounted_epr,discounted_epr_mean)

		#Variance
		discounted_epr_diff /= np.std(discounted_epr)
		#print(discounted_epr_diff,"standardized no human")
		#print(f"high: {max(discounted_epr)}, low: {min(discounted_epr)}")
		if not human_demonstration:
			epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
		else:
			scores = self.human_robot_agreement_score(robot_actions, human_actions, 4)
			#print(scores, "score")
			score_mean = np.mean(scores)
			score_diff = np.subtract(scores,score_mean)
			score_diff /= np.std(scores)
			#print(f"standardized human", score_diff)
			epdlogp *= score_diff*discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
		
		#print(f"probs: {epdlogp}")
		
		start_time = time.time()
		grad = self.policy_backward(eph, epdlogp)
		#print("--- %s seconds for policy backward ---" % (time.time() - start_time))
		
		for k in self._model: self._grad_buffer[k] += grad[k] # accumulate grad over batch


	#Saves an array that represents the human demonstration "distribution"
	def save_human_action(self, human_action):
		x = [0,0,0,0]
		x[human_action] = 1.0
		
		self.h_actions.append(x)

	def human_robot_agreement_score(self, a_robot, a_human, num_actions):
		score = 0
		score_list = []

		for i in range(len(a_robot)):
			score = 0
			for j in range(num_actions):
				if a_human[i][j] == 1.0:
					score += 1.0 - abs(a_human[i][j] - a_robot[i][j])
				else:
					score += -1 * a_human[i][j]
			score_list.append(score)

		return np.vstack(score_list)


	# called to update model parameters, generally every N episodes/games for some N
	def update_parameters(self):
		for k,v in self._model.items():
			g = self._grad_buffer[k] # gradient
			self._rmsprop_cache[k] = self._decay_rate * self._rmsprop_cache[k] + (1 - self._decay_rate) * g**2
			self._model[k] -= self._learning_rate * g / (np.sqrt(self._rmsprop_cache[k]) + 1e-5)
			self._grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
        
# this function should be called when an episode (i.e., a game) has finished
	def finish_human_episode(self):
		# stack together all inputs, hidden states, action gradients, and rewards for this episode
		
		a_robot = [0.9,0.1,0,0]
		a_human = [1,0,0,0]
		num_actions = 4
		print(f"Robot: {a_robot}, Human: {a_human}")
		print(f"Good: {self.human_robot_agreement_score(a_robot, a_human, num_actions)} \n")


		a_robot = [0,0.1,0,0.9]
		a_human = [0,0,1,0]
		num_actions = 4
		print(f"Robot: {a_robot}, Human: {a_human}")
		print(f"Bad: {self.human_robot_agreement_score(a_robot, a_human, num_actions)} \n")


		a_robot = [1,0,0,0]
		a_human = [1,0,0,0]
		num_actions = 4
		print(f"Robot: {a_robot}, Human: {a_human}")
		print(f"Perfect: {self.human_robot_agreement_score(a_robot, a_human, num_actions)}\n")

		a_robot = [0.5,0.3,0,0.2]
		a_human = [0,0,1,0]
		num_actions = 4
		print(f"Robot: {a_robot}, Human: {a_human}")
		print(f"Bad: {self.human_robot_agreement_score(a_robot, a_human, num_actions)}\n")

		a_robot = [0.3, 0.7, 0, 0.1]
		a_human = [0,0,0,1]
		num_actions = 4
		print(f"Robot: {a_robot}, Human: {a_human}")
		print(f"Bad: {self.human_robot_agreement_score(a_robot, a_human, num_actions)}\n")

		# this needs to be stored to be used by policy_backward
		# self._xs is a list of vectors of size input dim and the number of vectors is equal to the number of time steps in the episode
		self._epx = np.vstack(self._xs)
		#print(f"epx: {self._epx}, shape: {np.shape(self._epx)}")
		#for i in range(0,len(self._hs)):
		#	print(self._hs[i])
		
		# len(self._hs) = # time steps
		# stores hidden state activations
		eph = np.vstack(self._hs)
		# print(f"eph: {eph} shape: {np.shape(eph)}")
		# print(f"hs: {self._hs} shape: {np.shape(self._hs)}")

		
		#for i in range(0,len(self._dlogps)):
		#	print(self._dlogps[i])
		
		# self._dlogps stores a history of the probabilities over actions selected by the agent
		epdlogp = np.vstack(self._dlogps)
		#print(f"epdlogp: {epdlogp}, shape: {np.shape(epdlogp)}")
		
		self.human_advantage(epdlogp, self.h_actions)

		# self._drs is the history of rewards
		#for i in range(0,len(self._drs)):
		#	print(self._drs[i])
		epr = np.vstack(self._drs)
		#print(f"epr: {epr}")
		#print(f"epr shape: {np.shape(epr)}")
		

		self._xs,self._hs,self._dlogps,self._drs, self.h_actions, self.a_probs = [],[],[],[],[],[] # reset array memory

		# compute the discounted reward backwards through time
		discounted_epr = (self.discount_rewards(epr))
		#print(f"discounted_epr: {discounted_epr}")
		#print(f"disepr shape: {np.shape(discounted_epr)}")

		#for i in range(0,len(discounted_epr)):
		#	print(str(discounted_epr[i]) + "\t"+str(epr[i]))
		
		
		#print(discounted_epr)
		discounted_epr_mean = np.mean(discounted_epr)
		#print(f"discounted_epr_mean: {discounted_epr_mean}")

		#print(discounted_epr_mean)
		
		# standardize the rewards to be unit normal (helps control the gradient estimator variance)
		
		#discounted_epr -= np.mean(discounted_epr)
		discounted_epr = np.subtract(discounted_epr,discounted_epr_mean)
		
		
		discounted_epr /= np.std(discounted_epr)
		
		epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
		#print(f"probs: {epdlogp}")
		
		start_time = time.time()
		grad = self.policy_backward(eph, epdlogp)
		#print("--- %s seconds for policy backward ---" % (time.time() - start_time))
		
		for k in self._model: self._grad_buffer[k] += grad[k] # accumulate grad over batch


		
		
#Make one hot encoding vector of actions from human
#1 if perfect agree, -1 if perfectly disagree, other if prob is 1 and 0.76 for example
# take discounted rewards, normalize with mean and std
#make a separate function for human and autonomous finish_epsiode() types

"""
J:
Create a one hot encoding of the actions from the user. 
Use the equation jivko gave on the whiteboard to create the advantage
Get the discounted rewards for the USER, normalize it with the mean and std
Do I use the discounted reward to compute the advantage or the equation jivko gave?
multiply the (autonomous) probabilities by the advantage
"""
