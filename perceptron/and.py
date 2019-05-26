#  This program will complete the logic compute And
import numpy as np

class Perception:
    def __init__(self):
        self.w =  np.array([[0],[0]])
        self.b = 0


    def neuro(self, inputX):
        """single neuro cell
        
        [description]
        
        Arguments:
            inputX 2Dimension row vector -- as the input of perception
            return bool -- output after operation
        """ 
        if (np.dot(inputX, self.w) + self.b)[0] > 0:
            return 1
        else:
            return 0

    def train(self, inputX, labelY, rate=0.1):
        """train the model
            it will change the value of w and b
        Arguments:
            inputX 2Dimension row vector -- input of two operand
            labelY bool -- label of the sample inputX
        """
        outputY = self.neuro(inputX)
        self.w = self.w +  np.dot(rate*(labelY - outputY ),np.reshape(inputX,(2,-1)))
        self.b = self.b +  rate*(labelY - outputY)
            



perp = Perception()
stop=False
while stop == False:
    perp.train(np.array([0,0]), 0)
    perp.train(np.array([0,1]), 0)
    perp.train(np.array([1,0]), 0)
    perp.train(np.array([1,1]), 1)
    
    if perp.neuro(np.array([0,0])) == 0 and \
    perp.neuro(np.array([1,0])) == 0 and \
    perp.neuro(np.array([0,1])) == 0 and \
    perp.neuro(np.array([1,1])) == 1 :
        stop = True;

print('w = ' , perp.w)
print('b = %d' % perp.b)
print('0 and 0 = %d'% perp.neuro(np.array([0,0])))
print('1 and 0 = %d'%perp.neuro(np.array([1,0])))
print('0 and 1 = %d'%perp.neuro(np.array([0,1])))
print('1 and 1 = %d'%perp.neuro(np.array([1,1])))
