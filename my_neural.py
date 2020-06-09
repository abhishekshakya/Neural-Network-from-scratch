import numpy as np 
import matplotlib.pyplot as plt
plt.style.use("seaborn")

#-------------------------------------------------------------------------------Neural Network------------------------------------------------------

class NeuralNetwork:

	def __init__(self,input_size,layers,output_size):

		np.random.seed(0)
		self.layers_hidden = len(layers)
		self.output_size = output_size

		model = {}
		num = 2

		#we need to define weights matrix and bias vector
		#--------size of weight matrix of lth layer = (no of neurons in (l-1)th layer x no of neurons in (l)th layer)
		model['W1'] = np.random.randn(input_size,layers[0]);#layer one
		model['b1'] = np.zeros((1,layers[0]))

		for i in range(1,self.layers_hidden):
			model[f'W{i+1}'] = np.random.randn(layers[i-1],layers[i])#hidden layers
			model[f'b{i+1}'] = np.zeros((1,layers[i]))


		model[f'W{len(layers)+1}'] = np.random.randn(layers[len(layers)-1],output_size);#output layer
		model[f'b{len(layers)+1}'] = np.zeros((1,output_size))

		self.model = model
		return

	def forward_propagation(self,a):#     a is a list
		a = np.array(a)
		a = a.reshape(1,-1)
		
		self.model['a0'] = a#(1 x l-1)


		for i in range(1,self.layers_hidden+2):
			weight_matrix = self.model[f'W{i}']#(l-1 x l)
			bias_vector = self.model[f'b{i}']
			
			z = np.dot(self.model[f'a{i-1}'],weight_matrix) + bias_vector#(z=1 x l)
			if i!=self.layers_hidden+1:
				self.model[f'a{i}'] = self.activation(z)
			else:
				self.model[f'a{i}'] = self.softmax(z)
			
		y_ = self.model[f'a{self.layers_hidden+1}']
		return y_


	def activation(self,v):
		return np.tanh(v)#for now activation is tanh

	def loss(self,pred,y_real):
		l = -np.mean(y_real*np.log(pred))
		return l

	def softmax(self,v):
		ep = np.exp(v)
		ans = ep/np.sum(ep)
		return ans

	def backpropagation(self,pred,y_real,n=0.01):
		#doing back propagation for one example at a time

		delta = pred - y_real
		self.model[f'b{self.layers_hidden+1}'] -= (n*delta)
		self.model[f'W{self.layers_hidden+1}'] -= n*(np.dot(self.model[f'a{self.layers_hidden}'].T,delta))

		for j in range(self.layers_hidden,0,-1):
			delta1 = np.dot(delta,self.model[f'W{j+1}'].T)*((1-(self.model[f'a{j}'])**2))
			self.model[f'b{j}'] -= (n*delta1)
			self.model[f'W{j}'] -= n*(np.dot(self.model[f'a{j-1}'].T,delta1))
			delta = delta1


	def predict(self,X):
		pred =[]
		for i in range(X.shape[0]):
			y = self.forward_propagation(X[i])
			pred.append(np.argmax(y))
		return np.array(pred)

	def train(self,X,y_real,epoches=50,learning_rate=0.01):
		loss = []
		classes = self.output_size

		for i in range(epoches):
			l = 0.0
			for j in range(X.shape[0]):
				x = X[j]

				pred = self.forward_propagation(x)#(1 x l)
				l += self.loss(pred,y_real[j])

				self.backpropagation(pred,y_real[j],learning_rate)


			l = l/X.shape[0]
			loss.append(l)
			print(f'epoch {i} loss: {l}')


		return loss


def one_hot(y,classes):
		y_real = np.zeros((y.shape[0],classes))
		for i in range(y.shape[0]):
			output = y[i]
			y_real[i,output]=1
		return y_real


#==============================================================================================================================================

#----------------------------------------------------------------------Testing------------------------------------------------------------------

#------------------------------------------------------------------------------------
def main1():
	from sklearn.datasets import make_circles,make_classification
	X,Y = make_circles(n_samples=500, shuffle=True, noise=0.2, random_state=1, factor=0.2)
	y_real = one_hot(Y,len(np.unique(Y)))
	plt.scatter(X[:,0],X[:,1],c=Y)
	plt.show()


	NN = NeuralNetwork(input_size=2,layers=[10,5],output_size=2)
	loss = NN.train(X,y_real,epoches=500,learning_rate=0.001)

	plt.plot(loss,label='loss curve')
	plt.legend()
	plt.show()
	pred = NN.predict(X)
	print("accuracy",(np.sum(pred==Y)/X.shape[0]) * 100)


	#from matplotlib documentation---
	from visualize import plot_decision_boundary
	plot_decision_boundary(lambda x:NN.predict(x),X,Y)
	#------


if __name__ == '__main__':
	main1()








