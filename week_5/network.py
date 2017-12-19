import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from time import time
import pickle


def load(file):
	nn = NeuralNetwork((100,))
	saved = pickle.load(open(file, 'rb'))
	nn._weights = saved['weights']
	nn._bias = saved['bias']
	nn._hidden_layers = saved['hidden_layers']
	nn._alpha = saved['alpha']
	nn._epoch = saved['epochs']
	return nn

def non_linear(x):
	return 1 / (1 + np.exp(-x))

def sigmoid(x, deriv=False):
	"""
	Sigmoidal Activation function
	
	Args:
	 - x: vector or scaler, input to the sigmoid
	 - deriv: boolean to specify if we want to use sigmoid or its derivative
	 
	 Returns:
	 - y: scalar or vector, output of the sigmoid
	"""
	if deriv:
		x = non_linear(x)
		return x * (1 - x)
	return non_linear(x)

def visualize_image(X, y, pred):
	rgb = X
	img = rgb.reshape(3, 32, 32).transpose([1, 2, 0])
	plt.figure(figsize=(3, 2))
	plt.imshow(img)
	plt.title('label = {} / prediction = {}'.format(y, pred))
	plt.show()

class NeuralNetwork(object):
	def __init__(self, hidden_layers, alpha=0.001, epoch=8000, activation=sigmoid, optimiser='gradient', error='SSE', random_state=0):
		"""
		Instantiation for a neural network
		Only the full batch training is supported

		Args:
			- hidden_layers, tuple: each element of the tuple represents the number
			of hidden neurons in that index hidder layer
			- alpha, float: learning rate
			- epoch, int: number of training steps
			- activation, callable: activation function, only sigmoid is supported here
			- optimiser, string: algorithm of optimisation, only gradient descent supported here
			- error, string: defines the cost function, only the Summed Squared Error is supported here
			- random_state, int: defines the randomness of the model
		Returns:
			- NeuralNetwork object: trainable ANN
		"""
		self._hidden_layers = hidden_layers
		self._alpha = alpha
		self._epoch = epoch
		self._activation = activation
		self._seed = random_state

	def _create_network(self, n_features, n_outputs):
		"""
		Function to create the architecture of the neural net

		Args:
			- n_features, int: number of features/ neurons in input layer
			- n_outputs, int: number of neurons in output laye
		"""
		assert isinstance(self._hidden_layers, (int,tuple))

		try:
			weights_shapes = [n_features] + list(self._hidden_layers) + [n_outputs]
		except:
			weights_shapes = [n_features, self._hidden_layers, n_outputs]
		weights_shapes = [(weights_shapes[i],
						   weights_shapes[i+1]) for i in range(len(weights_shapes) - 1)]

		self._weights_shapes = weights_shapes

		def generate(shape):
			np.random.seed(self._seed)
			result = 2 * np.random.random(shape) - 1
			self._seed += 1
			return result

		self._weights = [generate(shape) for shape in weights_shapes]
		self._bias = [0 for shape in weights_shapes]
		self._cost = []
		self._train_curve = []
		self._test_curve = []

	def _forward_layer(self, X, w, b, f):
		"""
		forward propagate a single synapse

		Args:
			- X, matrix: input matrix of the current synapse
			- w, matrix: weights of the current synapse
			- b, scalar: bias of the current synapse
			- f, callable: activation functon to be used for current neurons
		Returns:
			- y: matrix, ouput of current synapse
		"""
		return f(np.dot(X, w) + b)

	def _forward(self, X):
		"""
		Function that implements the forward propagation
		
		Args:
			- X: input matrix of shape (m, n)
			
		Returns:
			- y: output matrix of shape (m, l)
		"""
		self._layers_output = []
		_input = np.copy(X)
		for w_, b_ in zip(self._weights, self._bias):
			_input = self._forward_layer(_input, w_, b_, self._activation)
			self._layers_output.append(_input)

		self._layers_output = list(reversed(self._layers_output))
		return _input

	def _eval(self, y_true, y_pred, deriv=False):
		"""
		Evaluation function

		Args:
			- y_true: vector, true labels
			- y_pred: vector, predicted probabilities
			- deriv: if we want to return the derivative or the error
		Returns:
			- result: error or its derivative
		"""
		y_true = y_true.reshape(-1, 1)
		y_pred = y_pred.reshape(-1, 1)
		if deriv:
			return (y_true - y_pred)
		return ((y_true - y_pred)**2) / 2.0

	def _backward_layer(self, error, alpha, X, w, b):
		"""
		back propagate a single synapse
		
		Args:
			- y_true: scalar or vector, used to calculate error term
			- y_pred: scalar or vecotr, output of the forward propagation
			- alpha: float, learning rate
			- X: matrix, input of the synapse
			- w: matrix, weights of the synapse
			- b: scalar or vector, bias of the synapse
			
		Returns:
			- w: matrix, updated weights
			- b: vector or scalar: updated bias
		"""
		h = np.dot(X, w) + b
		# print('step')
		# print(w.shape)
		# print(X.shape)
		# print(error.shape)
		delta = error * self._activation(h, True)

		derivative_w = - X.T.dot(delta)
		direction_w = - derivative_w
		
		derivative_b = - error.sum(axis=0, keepdims=True)
		direction_b = - derivative_b

		# print(X.shape)
		# print(h.shape)
		# print(error.shape)
		# print((error * self._activation(h, True)).shape)
		# print(derivative_w.shape)
		# print(w.shape)
		
		w += alpha * direction_w
		b += alpha * direction_b

		return w, b, delta

	def _backward(self, y_true, y_pred, alpha, X):
		"""
		Function that implements the back propagation

		Args:
			- y_true: vector, true targets
			- y_pred: vector, predicted probablities of the ANN
			- alpha: learning rate
			- X: matrix, input of the ANN
		"""
		_weights = list(reversed(self._weights))
		_bias = list(reversed(self._bias))
		_layers_output = self._layers_output
		_layers_input = _layers_output[1:] + [X]

		error = self._eval(y_true, y_pred, True)
		for i, (w_, b_, out_, in_) in enumerate(zip(_weights, _bias, _layers_output, _layers_input)):
			_weights[i], _bias[i], delta= self._backward_layer(error,
															   alpha,
															   in_,
															   w_,
															   b_)
			error = delta.dot(w_.T)

		self._weights = list(reversed(_weights))
		self._bias = list(reversed(_bias))

	def train(self, X, y, X_test, y_test):
		"""
		Training function of the ANN

		Args:
			- X: input matrix
			- y: output vector
		
		"""
		X = np.copy(X)
		y = np.copy(y)

		n_features = X.shape[1]
		n_outputs = 1
		self._create_network(n_features, n_outputs)

		for i in range(self._epoch):
			# make prediction
			y_pred = self._forward(X)

			# backpropagation using gradient descent
			self._backward(y, y_pred, self._alpha, X)

			# save the cost at this training step
			cost = self._eval(y, y_pred).sum()

			train_pred = self.predict(X)
			test_pred = self.predict(X_test)
			train_acc = accuracy_score(train_pred, y)
			test_acc = accuracy_score(test_pred, y_test)

			print('Epoch: {} / {} | Train accuracy {:.4f} | Test accuracy {:.4f} | Cost = {:.2f}'.format(i+1,
																							     self._epoch,
																							     train_acc,
																							     test_acc,
																							     cost), end='\r')

			self._train_curve.append(train_acc)
			self._test_curve.append(test_acc)
			self._cost.append(cost)
		print()

	def plot_error(self):
		"""
		Basic plot function to visualize the learning curve
		"""
		plt.figure(figsize=(16, 12))
		plt.plot(range(self._epoch), self._cost)
		plt.title('learning rate: {}'.format(self._alpha))
		plt.ylabel('SSE')
		plt.xlabel('Epoch')
		fig_name = input('Under what name would you like to save the learning curve figure ?\n')
		print('saving the figure under {}'.format(fig_name))
		plt.savefig(fig_name)
		print('displaying the saved learning curve')
		plt.show()

	def plot_train_test(self):
		plt.figure(figsize=(16, 12))
		plt.plot(range(self._epoch), self._train_curve, color='blue', lw=3, label='Train set curve')
		plt.plot(range(self._epoch), self._test_curve, color='red', lw=3, label='Test set curve')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		fig_name = input('Under what name would you like to save the train and test accuracy curves ?\n')
		print('saving the figure under {}'.format(fig_name))
		plt.savefig(fig_name)
		print('displaying the saved train and test accuracy curves')
		plt.show()
		
	def predict(self, X):
		"""
		Prediction function of the Network which rounds the predicted probability
		
		Args:
			- X: matrix to predict
		Returns:
			- y_pred: predicted outputs as an array of binary values
		"""
		y_pred = self._forward(X)
		return np.round(y_pred).astype(int)

	def predict_as_str(self, X):
		"""
		Simply converts the output to the corresponding label
		"""
		y_pred = self.predict(X)
		return ['cat' if pred == 1 else 'no cat' for pred in y_pred]

	def save(self, file):
		pickle.dump({'bias':self._bias, 'weights':self._weights,
					 'hidden_layers':self._hidden_layers,'alpha': self._alpha,
					 'epochs': self._epoch}, open(file, 'wb'))
		print('Neural network saved under {}'.format(file))


def main():
	print('Initializing..')
	sns.set()

	print('Loading data..')
	data = pd.read_csv('data.csv')
	mapping = {'cat':1, 'no_cat':0}
	data['target_n'] = data['target'].map(mapping)

	print('Shuffling and extracting inputs and targets')
	data = shuffle(data, random_state=42)
	X = data.drop(['target', 'target_n'], axis=1).astype('uint8')
	y = data['target_n']

	print('Preprocessing the data..')
	scaler = StandardScaler()
	X_scl = scaler.fit_transform(X)

	X_train, X_test, y_train, y_test = train_test_split(X_scl, y, test_size=0.2, random_state=0)

	print("------------------------------------\nWelcome to ENSTABrain's interactive Neural Network\n------------------------------------")
	mode = int(input("What do you wish to do ?\nPlease select the corresponding number from this menu:\n"+\
		"1- Train the network.\n2- Test the network on test set.\n3- Live testing on pictures\n0- Leave\n"))

	while mode != 0:
		if mode == 1:
			print('please sepcify the hyper parameters')
			hidden_layers = input('hidden_layers: Number of neurons in each hidden layer. \nExamples:\n'+\
								  '- (100,) for 1 hidden layer with 100 neurons\n'+\
								  '- (50, 70) for 2 hidden layers with 50 and 70 neurons respectively\n'+\
								  '- etc.\n')

			hidden_layers = hidden_layers.replace(r'(', '')
			hidden_layers = hidden_layers.replace(r')', '')

			hidden_layers = tuple(int(x.strip()) for x in hidden_layers.split(',') if x != '')

			alpha = float(input('alpha: Learning rate, float. \n Example: 0.001\n'))
			epochs = int(input('epochs: Number of training steps, int. \n Example: 1000\n'))

			model = NeuralNetwork(hidden_layers=hidden_layers, alpha=alpha, epoch=epochs, activation=sigmoid)

			start = time()
			print('Starting training..')
			model.train(X_train, y_train, X_test, y_test)
			end = time()
			print('training took {:.2f}s'.format(end-start))

			model_dir = 'models/'
			file_name = input('where do you wish to save the model ?\n')
			try:
				load(file_name)
				print('found existing neural network with name {}, do you want to overwrite it?[y/n]\n'.format(file_name))
				choice = input() 
				while choice not in ['y', 'n']:
					print('please select a correct answer [y/n]')
					choice = input()

				if choice == 'y':
					model.save(model_dir + file_name)
				else:
					file_name = input('please choose an alternative name or type "no" to not save the model\n')
					if file_name != 'no':
						model.save(model_dir + file_name)

			except:
				model.save(model_dir + file_name)
			print('plotting learning curve:')
			model.plot_error()
			print('plotting train and test accuracy curves')
			model.plot_train_test()

		elif mode == 2:
			try:
				file_name = input('please enter the name of the saved neural network\n')
				model = load(file_name)

				y_pred = model.predict(X_train)
				print('Train Accuracy:', accuracy_score(y_train, y_pred))

				y_pred = model.predict(X_test)
				print('Test Accuracy:', accuracy_score(y_test, y_pred))

			except:
				print('no saved model of the name {} found, you may want to train a neural network first'.format(file_name))


		elif mode == 3:
			#try:
			file_name = input('please enter the name of the saved neural network\n')
			model = load(file_name)
			option = int(input('Please enter the image index (<12000) to make a prediction on (type -1 to go back to last menu):\n'))

			while option != -1:
				image = X.values[option]
				label = data['target'].values[option]
				pred = model.predict_as_str(X_scl[option])
				visualize_image(image, label, pred)

				option = int(input('Please enter the image index (<12000) to make a prediction on (type -1 to go back to last menu):\n'))
			#except:
			#	print('no saved model of the name {} found, you may want to train a neural network first'.format(file_name))
		else:
			raise ValueError('option not known')

		mode = int(input("What do you wish to do ?\nPlease select the corresponding number from this menu:\n"+\
		"1- Train the network.\n2- Test the network on test set.\n3- live testing on pictures\n0- Leave\n"))

	print('Thank your for trying our Network, hope to see your again ;)')


# program start
if __name__ == '__main__':
	main()