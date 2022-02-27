import numpy as np
from numba import vectorize, cuda, float32, guvectorize, jit
import pickle
from os import path


class AI:
  e = np.float32(np.e)
  Layers = []
  Connections = []
  dtype = np.float32
  target = 'cpu'  # cuda, parallel, cpu
  '''
  def neuronF(self,x):
    return 1/(1+self.e**-x)
  def dneuronF(self,x):
    return self.neuronF(x)*(1-self.neuronF(x))
  '''

  @vectorize(['float32(float32, float32)'], target=target)
  def add(a, b):
    a = a + b
    return a

  @guvectorize([(float32[:, :], float32[:, :], float32[:, :])], '(m,l),(l,n)->(m,n)', target=target)
  def matmul_gu(A, B, out):
    """Perform square matrix multiplication of out = A * B
    """
    for i in range(out.shape[0]):
      for j in range(out.shape[1]):
        tmp = 0.
        for k in range(A.shape[1]):
          tmp += A[i, k] * B[k, j]
        out[i, j] = tmp

  @guvectorize([(float32[:, :], float32[:, :], float32, float32[:, :], float32[:, :])], '(l,m),(l,n),(),(m,n)->(m,n)',
               target=target)
  def modify_connections(errorDelta, backlayerResults, delta, connections, out):
    """Perform square matrix multiplication of out = transposed errorDelta * backlayerResults
    """

    # errorDelta=np.transpose(errorDelta)
    for i in range(out.shape[0]):
      for j in range(out.shape[1]):
        # tmp = 0.
        # for k in range(errorDelta.shape[1]):
        # tmp += errorDelta[i, k] * backlayerResults[k, j]
        out[i, j] = (errorDelta[0, i] * backlayerResults[0, j]) * delta + connections[i, j]
    # out=connections+out

    ####################################3
    # return out

  @guvectorize([(float32[:, :], float32[:, :], float32[:, :])], '(m,l),(n,l)->(m,n)',
               target=target)
  def propagate_errors(errorDelta, connections, out):
    """Perform square matrix multiplication of out = errorDelta * transposed connections
    """
    # newErrorSum=0.
    # for i in range(out.shape[0]):
    for j in range(out.shape[1]):
      tmp = 0.
      for k in range(connections.shape[1]):
        tmp += errorDelta[0, k] * connections[j, k]
      out[0, j] = tmp  # *actualLayerValue[0,j]*(1-actualLayerValue[0,j])
      # newErrorSum+=abs(tmp)
    # errorSum=0.
    # for x in errorDelta[0]:
    #  errorSum+=abs(x)

    # multiplier=errorSum*out.shape[1]/newErrorSum/errorDelta.shape[1]

    # for i in range(0,out.shape[1]):
    #  out[0,i]=multiplier*out[0,i]
    # for n in range(actualLayerValue.shape[1]):
    #  out[0,n]*=actualLayerValue[0,n]*(1-actualLayerValue[0,n])#multiply error * drerivative from actuallayervalue

    ########################################
    # return out

  @vectorize(['float32(float32)'], target=target)
  def _neuronF(x):
    x = 1 / (1 + float32(np.e) ** -x)
    return x

  '''
    def _neuronF(self,x):
    for i in range(x.shape[1]):
      if x[0,i]>86 or x[0,i]<-86:
        if x[0,i]>86:
          x[0, i]=float32(86)
        else:
          x[0, i] = float32(-86)
      x[0,i] = 1. / (1. + (self.e ** (-x[0,i])))
    return x
  '''

  @vectorize(['float32(float32)'], target=target)
  def _dneuronF(x):
    x = 1 / (1 + float32(np.e) ** -x)
    x = x * (1 - x)
    return x

  '''
  @vectorize(['float32(float32)'], target=target)
  def _dneuronF2(neuronF):
    neuronF = neuronF * (1 - neuronF)
    return neuronF'''

  @vectorize(['float32(float32,float32)'], target=target)
  def _dError(neuronF, error):
    error = neuronF * (1 - neuronF) * error
    return error

  @vectorize(['float32(float32, float32)'], target=target)
  def sub(a, b):
    a = a - b
    return a

  def cacheData(self):
    with open('cache.bin', 'wb') as _file:
      content = [
        self.exampleInputs,
        self.exampleOutputs,
        self.numberOfExamples,
        self.exampleInputst,
        self.exampleOutputst,
        self.numberOfExamplest]
      pickle.dump(content, _file)

  def __init__(self, _layers, step=0.):
    if not step == 0.:
      self.step = step

    if path.isfile('cache.bin'):
      with open('cache.bin', 'rb') as _file:
        content = pickle.load(_file)
        self.exampleInputs = content[0]
        self.exampleOutputs = content[1]
        self.numberOfExamples = content[2]
        self.exampleInputst = content[3]
        self.exampleOutputst = content[4]
        self.numberOfExamplest = content[5]

    for i in range(0, len(_layers)):
      newVec = np.empty(shape=(1, _layers[i]), dtype=self.dtype)
      # newVec=np.random.random_sample(_layers[i],)

      # for i in range(0,_layers[i]):
      #  newVec[0][i]=np.random.random_sample()*12-6
      self.Layers.append(newVec)

    for i in range(0, len(self.Layers) - 1):
      connectionMatrix = np.empty(shape=(self.Layers[i + 1].size, self.Layers[i].size), dtype=self.dtype)
      randomRange = .2  # 1.
      np.random.seed(seed=2)
      for m in range(0, self.Layers[i + 1].size):
        for n in range(0, self.Layers[i].size):
          connectionMatrix[m][n] = np.random.random_sample() * randomRange * 2. - randomRange
      self.Connections.append(connectionMatrix)
    '''
    print(self.neuronF(1))
    x=np.ones(shape=(1,1),dtype=self.dtype)
    print(self._neuronF(x))
    print(self.dneuronF(1))
    x = np.ones(shape=(1, 1), dtype=self.dtype)
    print(self._dneuronF(x))'''
    # self.desiredOutput=np.empty(shape=(1,_layers[0]),dtype=self.dtype)
    print(self.Layers)
    # print(self.Connections)
    # self.setInput()
    # self.flow()

    # self.readExamplesFile()
    # self.teach()

    # self.getConnectionsFromFile()
    # self.setInput()
    # self.flow()
    '''
    a=np.ones(shape=(1,5),dtype=np.float32)
    b = np.ones(shape=(1, 5), dtype=np.float32)
    self.sub(a,b)
    print(a)'''

  def setInput(self):
    lastLayerIndex = len(self.Layers) - 1
    for i in range(0, self.Layers[lastLayerIndex].size):
      self.Layers[lastLayerIndex][0, i] = 0.5
    # print(self.Layers[lastLayerIndex])

  # actualInputOutput=0
  index = 0

  def setInputOutputt(self, number):
    # output = self.exampleOutputs[number]
    lastLayerIndex = len(self.Layers) - 1
    self.Layers[lastLayerIndex] = self.exampleInputst[number]
    return self.exampleOutputst[number]

  def setInputOutput(self, number):
    # output = self.exampleOutputs[number]
    lastLayerIndex = len(self.Layers) - 1
    self.Layers[lastLayerIndex] = self.exampleInputs[number]
    return self.exampleOutputs[number]
    '''
    #self.actualInputOutput+=1
    #self.actualInputOutput = 1
    #self.index+=1
    #self.index=self.index%3000
    #self.actualInputOutput=int(self.index/1000)
    #self.actualInputOutput=self.actualInputOutput%3
    self.actualInputOutput=number

    for i in range(0,3):
      output[0,i]=0
    output[0,self.actualInputOutput]=1
    lastLayerIndex = len(self.Layers) - 1
    for i in range(0, self.Layers[lastLayerIndex].size):
      self.Layers[lastLayerIndex][0, i] = 0.5
    if self.actualInputOutput is 0:
      for i in range(0,5):
        self.Layers[lastLayerIndex][0, i] = 0
    elif self.actualInputOutput is 1:
      for i in range(5,10):
        self.Layers[lastLayerIndex][0, i] = 1'''

  def flow(self):
    for i in range(len(self.Connections) - 1, -1, -1):
      self.matmul_gu(self.Layers[i + 1], self.Connections[i], self.Layers[i])
      # self.Layers[i]=self._neuronF(np.clip(self.Layers[i],float32(-10.),float32(10.)))
      self.Layers[i] = self._neuronF(self.Layers[i])

      # if np.isnan(self.Layers[i][0, 0]):
      #  br = 1

    # print(self.Layers[0])

  saveName = 'neurons.bin'
  # step=0.000000001#0.00000000005
  step = 0.000001

  # learningRate=0.01

  def teach(self):
    '''
    self.setInput()
    self.flow()'''
    # self.Layers[0]=

    desiredOutput = np.empty(shape=(1, self.Layers[0].size), dtype=self.dtype)
    # desiredOutput[0,0]=1
    # desiredOutput[0, 1] = 0
    # desiredOutput[0, 2] = 0

    errorVectors = []
    for vec in self.Layers:
      errorVectors.append(np.copy(vec))
    # print()
    # print(errorVectors)
    j = 0
    ConnChange = []
    for _i in range(0, len(self.Connections)):
      ConnChange.append(np.zeros(shape=self.Connections[_i].shape, dtype=self.dtype))
    bestPerformance = 0
    while True:
      # for j in range(0,10000):
      # ConnChange=[]
      for _i in range(0, len(self.Connections)):
        ConnChange[_i] *= 0
      indexes = np.random.permutation(self.numberOfExamples)
      for _o in range(0, self.numberOfExamples):
        o = indexes[_o]
        desiredOutput = self.setInputOutput(o)
        self.flow()
        j += 1
        '''
        if j>=100:
          j=0
          print(str(self.Layers[0]) + "   " + str(self.actualInputOutput))
        '''
        if (j >= 1 and j <= 10):
          for k in range(10):
            if self.exampleOutputs[o][0, k] > 0.5:
              break
          print(str(self.Layers[0]) + "      example number " + str(o) + "   " + str(k))
          # print(str(self.Layers[0])+"   "+str(desiredOutput))
        if j >= 70000:
          # print(self.Layers[0])
          j = 0
          # save connections
          self.save()
        # self.setInput()'''

        errorVectors[0] = self.sub(desiredOutput, self.Layers[0])
        # errorVectors[0]=errorVectors[0]

        # print(errorVectors[0])
        errorVectors[0] = self._dError(self.Layers[0], errorVectors[
          0])  # now errorVectors[0] equals derror/dx where x is sum(layer before*connections)
        # print(errordx)
        # print()
        # print(self.Connections[0])
        # self.Connections[0]=self.modify_connections(self.Layers[1],errorVectors[0],self.step,self.Connections[0],self.Connections[0])
        '''
        ConnChange[0]=self.modify_connections(self.Layers[1],errorVectors[0],self.step,ConnChange[0],ConnChange[0])
        #print(self.Connections[0])
        errorVectors[1]=self.propagate_errors(errorVectors[0],self.Connections[0],errorVectors[1])
        errorVectors[1] = self._dError(self.Layers[i + 1], errorVectors[i + 1])'''
        # print(errorVectors[0])
        # print(errorVectors[1])
        for i in range(0, len(self.Connections)):
          self.Connections[i] = self.modify_connections(self.Layers[i + 1], errorVectors[i], self.step,
                                                        self.Connections[i], self.Connections[i])
          # ConnChange[i]= self.modify_connections(self.Layers[i+1], errorVectors[i], self.step, ConnChange[i],ConnChange[i])
          # if np.isnan(ConnChange[i][0, 0]):
          #  b = 1
          # if np.isinf(ConnChange[i][0, 0]):
          #  b = 1
          errorVectors[i + 1] = self.propagate_errors(errorVectors[i], self.Connections[i], errorVectors[i + 1])
          # if np.isnan(errorVectors[i+1][0, 0]):
          #  b = 1
          # self.dneuronF(errorVectors[i+1])
          errorVectors[i + 1] = self._dError(self.Layers[i + 1], errorVectors[i + 1])
          # if np.isnan(errorVectors[i][0, 0]):
          #  b = 1
        '''
        normSum=0.
        for _i in range(0, len(self.Connections)):
          normSum+=np.linalg.norm(ConnChange[_i])
        if normSum == 0.:
          self.save()
          return
        for _i in range(0, len(self.Connections)):
          ConnChange[_i]*=self.step/normSum
        '''
        # for _i in range(0,len(self.Connections)):
        #  self.Connections[_i]=self.add(ConnChange[_i],self.Connections[_i])
        # if np.isnan(ConnChange[_i][0,0]):
        #  b=1
      print(self.step)
      performance = self.test()
      '''
      if performance>bestPerformance:
        bestPerformance=performance
        self.regress=0
      elif performance<bestPerformance:
        self.regress+=1
        if self.regress>3:
            self.step=self.step/5
            bestPerformance=performance
      '''

  # regress=0

  def save(self):
    with open(self.saveName, 'wb') as save_file:
      pickle.dump(self.Connections, save_file)

  def getConnectionsFromFile(self):
    with open(self.saveName, 'rb') as read_file:
      self.Connections = pickle.load(read_file)
      self.Layers = []
      i = 0
      for i in range(len(self.Connections)):
        self.Layers.append(np.zeros((1, self.Connections[i].shape[1]), dtype=self.dtype))
      self.Layers.append(np.zeros((1, self.Connections[i].shape[0]), dtype=self.dtype))

  numberOfExamples = 0
  exampleInputs = []
  exampleOutputs = []
  numberOfExamplest = 0
  exampleInputst = []
  exampleOutputst = []

  def readExamplesFile(self, images='train-images.idx3-ubyte', labels='train-labels.idx1-ubyte'):
    if self.numberOfExamples > 0:
      return
    with open(images, 'rb') as examples_file, open(labels, 'rb') as labels_file:
      order = 'big'
      size = 28
      int.from_bytes(examples_file.read(4), order)
      self.numberOfExamples = int.from_bytes(examples_file.read(4), order)
      print("Number of examples: " + str(self.numberOfExamples))
      print("size: " + str(int.from_bytes(examples_file.read(4), order)) + "x" + str(
        int.from_bytes(examples_file.read(4), order)))
      # self.numberOfExamples=10000
      print("reading examples...")
      for i in range(self.numberOfExamples):
        self.exampleInputs.append(np.empty(shape=(1, size * size), dtype=self.dtype))
        for j in range(size * size):
          self.exampleInputs[i][0, j] = int.from_bytes(examples_file.read(1), order) / 255
      # print(self.exampleInputs[0])
      st = ''
      for j in range(size * size):
        if j % size == 0:
          st += '\n'
        if self.exampleInputs[0][0, j] == 0.:
          st += '_'
        else:
          st += 'O'
      print(st)
      int.from_bytes(labels_file.read(8), order)
      print("reading labels...")
      for i in range(self.numberOfExamples):
        self.exampleOutputs.append(np.zeros(shape=(1, 10), dtype=self.dtype))
        self.exampleOutputs[i][0, int.from_bytes(labels_file.read(1), order)] = 1.
        # print(self.exampleOutputs[i])
        # print(int.from_bytes(labels_file.read(1), order))
      # print(self.exampleOutputs[0])
      print("finished reading labels")

  def treadExamplesFile(self, images='train-images.idx3-ubyte', labels='train-labels.idx1-ubyte'):
    if self.numberOfExamplest > 0:
      return
    with open(images, 'rb') as examples_file, open(labels, 'rb') as labels_file:
      order = 'big'
      size = 28
      int.from_bytes(examples_file.read(4), order)
      self.numberOfExamplest = int.from_bytes(examples_file.read(4), order)
      print("Number of examples: " + str(self.numberOfExamplest))
      print("size: " + str(int.from_bytes(examples_file.read(4), order)) + "x" + str(
        int.from_bytes(examples_file.read(4), order)))
      # self.numberOfExamples=10000
      print("reading examples...")
      for i in range(self.numberOfExamplest):
        self.exampleInputst.append(np.empty(shape=(1, size * size), dtype=self.dtype))
        for j in range(size * size):
          self.exampleInputst[i][0, j] = int.from_bytes(examples_file.read(1), order) / 255
      # print(self.exampleInputs[0])
      st = ''
      for j in range(size * size):
        if j % size == 0:
          st += '\n'
        if self.exampleInputst[0][0, j] == 0.:
          st += '_'
        else:
          st += 'O'
      print(st)
      int.from_bytes(labels_file.read(8), order)
      print("reading labels...")
      for i in range(self.numberOfExamplest):
        self.exampleOutputst.append(np.zeros(shape=(1, 10), dtype=self.dtype))
        self.exampleOutputst[i][0, int.from_bytes(labels_file.read(1), order)] = 1.
        # print(self.exampleOutputs[i])
        # print(int.from_bytes(labels_file.read(1), order))
      # print(self.exampleOutputs[0])
      print("finished reading labels")

  def test(self):
    good = 0
    all = 0
    for i in range(self.numberOfExamplest):
      # for i in range(min(len(self.exampleOutputs),len(self.exampleInputs))):
      output = 0
      for output in range(10):
        if self.exampleOutputst[i][0, output] > 0.9:
          break
      self.setInputOutputt(i)
      self.flow()
      maxValue = self.Layers[0][0, 0]
      maxInd = 0
      for k in range(1, 10):
        if self.Layers[0][0, k] > maxValue:
          maxValue = self.Layers[0][0, k]
          maxInd = k
      all += 1
      if maxInd == output:
        good += 1
    print("Accuracy " + str(good / all))
    return good / all
  def ask(self,vec):
    if vec.shape==self.Layers[len(self.Layers)-1].shape:
      self.Layers[len(self.Layers) - 1]=vec
      self.flow()
      sum=np.sum(self.Layers[0])
      self.Layers[0]=self.Layers[0]/sum
      #max=self.Layers[0]
      maxIndex=0
      for i in range(10):
        if self.Layers[0][0,i]>self.Layers[0][0,maxIndex]:
          maxIndex=i
      return maxIndex,np.copy(self.Layers[0])





if __name__ == '__main__':
  # AI=AI([3,5,7,9,11,13,15,17,19])
  #AI = AI([3, 6, 8, 10])
  AI = AI([10,450,450,784],0.0001)#784=28px*28px
  #AI=AI([])
  #AI.getConnectionsFromFile()

  '''
  AI.treadExamplesFile('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')
  AI.readExamplesFile()
  AI.cacheData()
  AI.teach()
  '''




  AI.getConnectionsFromFile()
  #AI.treadExamplesFile('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')
  #AI.test()
  from PIL import Image
  with Image.open("img.png",'r') as image:
    b = list(image.getdata())
    #b = bytearray(f)
    vec=np.zeros((1,28*28),AI.dtype)
    for i in range(0,28*28):
      vec[0,i]= 1.-(b[i][0]+b[i][1]+b[i][2])/3/255
    #print(vec)
    s=""
    for i in range(28*28):
      if i%28 == 0:
        s+='\n'
      if vec[0,i]>0.5:
        s+='OO'
      else:
        s+='--'
    print(s)

    answer,percentage=AI.ask(vec)

    for i in range(percentage.shape[1]):
      print(str(i)+":  "+str(percentage[0,i]*100)+"%")
    print("Answer: " + str(answer) + " (" + str(percentage[0, answer]*100) + "%)")






