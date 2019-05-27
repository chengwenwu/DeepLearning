# this file will implement a back propagation to recognize mnist data set
import numpy as np
class Bpropagation:
	#这里是创建一个网络只需要关注网络的相关信息，而不需要关注它的训练等信息。
	#由于我们这里已经确定了输入层和输出层的神经元个数，并且确定隐藏层只有1层
	#所以需要传递的参数就是隐藏层神经元数量
	def __init__(self, hiddenNeuroNum):
		self.p_hiddenNeuroNum = hiddenNeuroNum #记录隐藏层神经元数目
		self.inputNNum = 728 #记录输入层神经元数目
		self.outputNNum = 10 #记录输出层神经元数目
		self.data			 #记录用来训练或者测试的样本输入
		self.label			 #记录用来训练或者测试的标签输入
		self.modelPar1=np.zero(self.p_hiddenNeuroNum, self.inputNNum  + 1)				 #用来存放模型参数的矩阵，应该有两个
		self.modelPar2=np.zero(self.outputNNum,self.p_hiddenNeuroNum + 1)
		self.sample_loc
		self.label_loc

	def load_sample(self, trainOrTest, number):# 每次读进来number个,下次从上次停止的地方继续读number个
			


	def load_label(self, trainOrTest, number):
