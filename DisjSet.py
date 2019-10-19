class DisjSet:

	def __init__(self,N):
		self.S = []
		self.size = N
		for i in range(0,N):   # vertices are 0 .. N-1
			self.S.append(i)
	
	def find(self,v):
		if (v < 0) or (v >= self.size):				
			raise self.InvalidInput
		f = v
		while (self.S[v] != v):
			v = self.S[v]
		while (f != v): # Path compression
			g = self.S[f]
			self.S[f] = v
			f = g
		return(v)
	
	def merge(self,s,t):
		if (s < 0) or (t < 0) or (s >= self.size) or (t >= self.size):
			raise self.InvalidInput
		if (self.S[s] != s) or (self.S[t] != t):
			raise self.InvalidInput
		self.S[s] = t

	class InvalidInput(Exception):
		pass
