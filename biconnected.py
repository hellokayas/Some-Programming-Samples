def readGraph():
	line = input()
	l = line.split()
	N = int(l[0])
	M = int(l[1])
	G = []

	for i in range(0,N):
		G.append([])
		
	for i in range(0,M):
		line = input()
		l = line.split()
		ls = list(map(int,l))
		G[ls[0]].append(ls[1])
		G[ls[1]].append(ls[0])

	return G
			
def biconnected():
	G = readGraph()
	N = len(G)
	# Initialize articulation DFStree, pts and bridges to empty set
	ArPts = set()
	Bridges = set()
	Par = [-1]*N
	DFSNo = [-1]*N
	LowNo = [N]*N
	dfsno = 0

	# DFS with an argument to indicate if it is the root of a component
	def doDFS(v,Root):
		nonlocal dfsno,DFSNo,LowNo,Par,Bridges,ArPts,G,N
		noChil = 0
		DFSNo[v] = dfsno
		dfsno = dfsno+1
		LowNo[v] = DFSNo[v]
		for w in G[v]:
			if DFSNo[w] < 0: # Tree Edge
				Par[w] = v
				noChil = noChil + 1
				doDFS(w,False)
				if LowNo[w] >= DFSNo[v] and len(G[v]) > 1 and not Root:
					ArPts = ArPts | {v}
				if LowNo[w] > DFSNo[v] and v!= N:
					Bridges = Bridges | {(v,w)}
				LowNo[v] = min(LowNo[v],LowNo[w])
			elif Par[v] != w: # backedge
				LowNo[v] = min(LowNo[v],DFSNo[w])
		if Root and noChil > 1:
			ArPts = ArPts | {v}

	for i in range(0,N): # Explore all connected components
		if DFSNo[i] < 0 : # A new component
			doDFS(i,True)  # process this component
		
	print("Articulation Points: ",ArPts)
	print("Bridges: ",Bridges)
			

biconnected()
