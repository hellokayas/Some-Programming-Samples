class Queue():

## Naive Queue. does not reuse space.

	def __init__(self,ls=[]):
		self.q = ls[:]
		self.b = 0
		self.e = len(ls)

	def isempty(self):
		return (self.b == self.e)

	def enqueue(self,v):
		self.q.append(v)
		self.e = self.e + 1
	
	def dequeue(self):
		self.b = self.b + 1
		return(self.q[self.b - 1])
		
		
