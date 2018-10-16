import numpy as np

def sig(z):
    return 1/(1+np.exp(-z))

class layer:
    def __init__(self, inputs, height):
        self.weight = np.random.rand(height,inputs)
        self.biases = np.random.rand(height)
    
    def feed_fwd(self, x): #x - input vector
        print(x)
        o = np.dot(self.weight,x)+self.biases
        print(o)
        o = [sig(x) for x in o]
        return o

    def __str__(self):
        s = "W :"+ str(self.weight)
        s+= "\n"
        s+="B :"+ str(self.biases)
        return s
class nn:
    def __init__(self, arch):
        # self.layers -- lista warstw sieci
        self.layers = []
        for i in range(1,len(arch)):
            l = layer(arch[i-1],arch[i])
            print("Layer "+str(i))
            print(l)
            self.layers.append(l)

    def feed_fwd(self,x): #x - input vector
        o = x.copy()
        for l in self.layers :
            o = np.array(l.feed_fwd(o))
        return o

arch = [6,5,4,4,5]
net = nn(arch)
x = np.array([2,1,3,1,2,5])
o = net.feed_fwd(x)
print("input")
print(x)
print("output")
print(o)
                      
                      
         
