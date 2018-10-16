import numpy as np

def sig(z):
    return 1/(1+np.exp(-z))

class layer:
    def __init__(self, inputs, height, file=None):
        if file is None:
           self.weight = np.random.rand(height,inputs)
           self.biases = np.random.rand(height)
        else:
           temp_w = []
           for x in range(0,height):
               row = [float(x) for x in file.readline().split(';')]
               temp_w.append(row)
           self.weight = np.array(temp_w).reshape(height,inputs)
           row = file.readline().split(';')
           self.biases = np.array(row,dtype=float)        
    
    def feed_fwd(self, x): #x - input vector
        ##print(x)
        o = np.dot(self.weight,x)+self.biases
        ##print(o)
        o = [sig(x) for x in o]
        return o
    def to_file(self,file):
        for i in range(0,np.size(self.weight,0)):
            for j in range(0,np.size(self.weight,1)):
                if j>0 : file.write(';')
                file.write(str(self.weight[i][j]))
            file.write("\n")
        for i in range(0,len(self.biases)):
            if i>0 : file.write(';')
            file.write(str(self.biases[i]))
        file.write("\n")
    def __str__(self):
        s = "W :"+ str(self.weight)
        s+= "\n"
        s+="B :"+ str(self.biases)
        return s
    
class nn:
    def __init__(self, arch=None, filename=None):
        file=None
        if filename is None:
            if arch is None:
                print ("nn has to have architecture or file")
                raise Exception('nn has to have architecture or file')             
            self.arch = arch
        else:
            file = open(filename,mode='r')
            row = file.readline().split(';')
            self.arch = np.array(row,dtype=int)    
        # self.layers -- lista warstw sieci
        self.layers = []
        for i in range(1,len(self.arch)):
            l = layer(self.arch[i-1],self.arch[i],file)
            ##print("Layer "+str(i))
            ##print(l)
            self.layers.append(l)
        if file is not None:
            file.close()

    def feed_fwd(self,x): #x - input vector
        o = x.copy()
        for l in self.layers :
            o = np.array(l.feed_fwd(o))
        return o

    def save_to_file(self,filename):
        file = open(filename,mode='w')
        for i in range(0,len(self.arch)):
            if i>0 : file.write(';')
            file.write(str(self.arch[i]))
        file.write("\n")
        for layer in self.layers:
            layer.to_file(file)
        file.close()

arch = [6,5,4,4,5]
net = nn(arch)
x = np.array([2,1,3,1,2,5])
o = net.feed_fwd(x)
print("input")
print(x)
print("output")
print(o)
net.save_to_file("nn.txt")
net2 = nn(filename="nn.txt")
o2 = net2.feed_fwd(x)
print("output")
print(o)
print("output 2")
print(o2)
                      
                      
         
