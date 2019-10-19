class Tree:
	
	def __init__(self,v=None,l=None,r=None):
		self.value = v
		self.left = l
		self.right = r

	def isempty(self):
		if self.value == None:
			return True
		return False

	def isleaf(self):
		if self.isempty():
			return False
		if self.left or self.right:
			return False
		return True

	def __str__(self):	
		if self.isempty():
			return("Empty Tree")
		if self.isleaf():
			return(str(self.value))
		if self.left:
			lstr = self.left.__str__()
		else: 
			lstr = "_"
		if self.right:
			rstr = self.right.__str__()
		else:
			rstr = "_"
		return("("+str(self.value)+" "+lstr+" "+rstr+")")
