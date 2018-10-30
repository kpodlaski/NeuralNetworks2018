import numpy as np
import random
import math
# network based on http://neuralnetworksanddeeplearning.com

def sig(z):
    return 1 / (1 + np.exp(-z))

def sig_derivative(z):
    return sig(z)*(1-sig(z))


class layer:
    def __init__(self, inputs, height, file=None):
        if file is None:
            self.weight = np.random.rand(height, inputs)
            self.biases = np.random.rand(height)
        else:
            temp_w = []
            for x in range(0, height):
                row = [float(x) for x in file.readline().split(';')]
                temp_w.append(row)
            self.weight = np.array(temp_w).reshape(height, inputs)
            row = file.readline().split(';')
            self.biases = np.array(row, dtype=float)

    def summed_input(self, x):
        z = np.dot(self.weight, x) + self.biases
        return z

    def activation_from_summed_input(self,z):
        return np.array([sig(x) for x in z])

    def feed_fwd(self, x):  # x - input vector
        ##print(x)
        o = np.dot(self.weight, x) + self.biases
        ##print(o)
        o = [sig(x) for x in o]
        return o

    def to_file(self, file):
        for i in range(0, np.size(self.weight, 0)):
            for j in range(0, np.size(self.weight, 1)):
                if j > 0: file.write(';')
                file.write(str(self.weight[i][j]))
            file.write("\n")
        for i in range(0, len(self.biases)):
            if i > 0: file.write(';')
            file.write(str(self.biases[i]))
        file.write("\n")

    def __str__(self):
        s = "W :" + str(self.weight)
        s += "\n"
        s += "B :" + str(self.biases)
        return s


class nn:
    def __init__(self, arch=None, filename=None):
        file = None
        if filename is None:
            if arch is None:
                print("nn has to have architecture or file")
                raise Exception('nn has to have architecture or file')
            self.arch = arch
        else:
            file = open(filename, mode='r')
            row = file.readline().split(';')
            self.arch = np.array(row, dtype=int)
            # self.layers -- lista warstw sieci
        self.layers = []
        for i in range(1, len(self.arch)):
            l = layer(self.arch[i - 1], self.arch[i], file)
            ##print("Layer "+str(i))
            ##print(l)
            self.layers.append(l)
        if file is not None:
            file.close()

    def feed_fwd(self, x):  # x - input vector
        o = x.copy()
        for l in self.layers:
            o = np.array(l.feed_fwd(o))
        return o

    #Stochastic Gradient Descent training
    def train(self, training_data, epochs, mini_batch_size,eta):
        for epoch in range(0,epochs):
            random.shuffle( training_data)
            mini_batces = [ training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batces:
                self.minibatch_update(mini_batch,eta)
            print ("Training epoch {0} complete".format(epoch))

    def back_propagate_input(self, input, expected_output):
        delta_for_layers  = []
        for layer in self.layers:
            delta_b = [np.zeros(b.shape) for b in layer.biases]
            delta_w = [np.zeros(w.shape) for w in layer.weight]
            delta_for_layers.append({'biases':delta_b, 'weight':delta_w})
        activation = input
        activations=[input]
        z = []
        #feedforward
        for layer in self.layers:
            actual_z = layer.summed_input(activation)
            z.append(actual_z)
            activation = layer.activation_from_summed_input(actual_z)
            activations.append(activation)
        #going backward
        delta = self.cost_derivative(activations[-1],expected_output)* sig_derivative(z[-1])
        delta_for_layers[-1]['biases'] = delta
        delta_for_layers[-1]['weight'] = np.outer(delta,activations[-2])
        for layer in range (2,len(self.layers)+1):
            actual_z = z[-layer]
            s_der = sig_derivative(actual_z)
            delta = np.dot(np.transpose(self.layers[-layer+1].weight), delta)*s_der
            delta_for_layers[-layer]['biases']= delta
            delta_for_layers[-layer]['weight']= np.outer(delta,activations[-layer-1])
        return delta_for_layers


    def minibatch_update(self, mini_batch, eta):
        #print("Batch : "+ str(mini_batch))
        delta_for_layers = []
        for layer in self.layers:
            delta_b = [np.zeros(b.shape) for b in layer.biases]
            delta_w = [np.zeros(w.shape) for w in layer.weight]
            delta_for_layers.append({'biases':delta_b, 'weight':delta_w})
        for m in mini_batch:
            _in = m['signal']
            _exp = m['exp_value']
            grad_for_layer_elem = self.back_propagate_input(_in,_exp)
            #print("grad:"+str(grad_for_layer_elem))
            for layer in range(0,len(self.layers)):
                delta_for_layers[layer]['biases'] = delta_for_layers[layer]['biases']+\
                                                    grad_for_layer_elem[layer]['biases']


                delta_for_layers[layer]['weight'] = delta_for_layers[layer]['weight']+\
                                                    grad_for_layer_elem[layer]['weight']

        #print("delta:"+str(delta_for_layers))
        #Update weights and biases

        for layer in range(0,len(self.layers)):
            self.layers[layer].weight = self.layers[layer].weight -\
                                        (eta/len(mini_batch))*delta_for_layers[layer]['weight']

            self.layers[layer].biases = self.layers[layer].biases -\
                                        (eta / len(mini_batch)) * delta_for_layers[layer]['biases']

    def cost_derivative(self, output_activations, expected_output):
        #quadratic cost
        return np.array(output_activations - expected_output).transpose()


    def save_to_file(self, filename):
        file = open(filename, mode='w')
        for i in range(0, len(self.arch)):
            if i > 0: file.write(';')
            file.write(str(self.arch[i]))
        file.write("\n")
        for layer in self.layers:
            layer.to_file(file)
        file.close()



def test_main():
    _arch = [4, 5, 4, 3, 5]
    _net = nn(_arch)
    x = np.array([2, 1, 3, 1, 2, 5])
    o = _net.feed_fwd(x)
    #_net.save_to_file("nn.txt")
    #net2 = nn(filename="nn.txt")
    o2 = _net.feed_fwd(x)
    print("output")
    print(o)
    print("output 2")
    print(o2)


def test_2():
    arch = [2,2,2]
    net = nn(arch)
    net.layers[0].weight= np.array([[.15,.20],[.25,.3]])
    net.layers[1].weight = np.array([[.4, .45], [.5, .55]])
    net.layers[0].biases = np.array([.35,.35])
    net.layers[1].biases = np.array([.6, .6])
    print (net.layers[0].weight)
    print(net.layers[1].weight)
    print(net.layers[0].biases)
    print(net.layers[1].biases)
    _in = np.array([.05,.1])
    _exp = np.array([.01,.99])
    _out = net.feed_fwd(_in)
    train_set = list()
    train_set.append({"signal":_in,"exp_value":_exp})
    train_set.append({"signal": _in, "exp_value": _exp})
    train_set.append({"signal": _in, "exp_value": _exp})
    train_set.append({"signal": _in, "exp_value": _exp})
    train_set.append({"signal": _in, "exp_value": _exp})
    train_set.append({"signal": _in, "exp_value": _exp})
    print(train_set)
    net.train(train_set,10000,2,0.5)
    #net.train(train_set, 10000, 6, 0.5)
    #minibatch_update(train_set,0.5)
    print("OUTPUT")
    _out = net.feed_fwd(_in)
    print("w0"+str(net.layers[0].weight))
    print("w1"+str(net.layers[1].weight))
    print("b0"+str(net.layers[0].biases))
    print("b1"+str(net.layers[1].biases))
    print(_out)


test_2()


