def readGraph():
	line = input()
	l = line.split()
	N = int(l[0])
	M = int(l[1])
	G = [] # Graph

	for i in range(0,N):
		G.append([])
		
	for i in range(0,M):
		line = input()
		l = line.split()
		ls = list(map(int,l))
		G[ls[0]].append(ls[1])
	return (G)
			
def topsort(): # Output vertices in topologically sorted order with level number
	G = readGraph()
	N = len(G)
	indeg = [0]*N
	levelNo = [0]*N
	topOrder = []
	zero = []
	for i in range(0,len(G)): # compute indegrees
		for v in G[i]:
			indeg[v] = indeg[v]+1

	for v in range(0,N):# find indegree 0 vertices
		if (indeg[v] == 0): 
			levelNo[v] = 0
			zero.append(v)

	while zero != [] :
		v = zero.pop()
		for w in G[v]:
			indeg[w] = indeg[w] - 1
			# If you use a queue instead of a stack 
			# this max operation can be avoided
			levelNo[w] = max(levelNo[w],levelNo[v]+1)
			if (indeg[w] == 0):
				zero.append(w)
		topOrder.append(v)

	print("A Topologically Sorted Listing:", topOrder)
	print("Level Numbers")
	for i in range(0,N):
		print("\t",i, ":",levelNo[i])

topsort()
