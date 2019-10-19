def readGraph():
	line = input()
	l = line.split()
	N = int(l[0])
	M = int(l[1])
	G = [] # Graph
	GR = [] # Graph reversed edges

	for i in range(0,N):
		G.append([])
		GR.append([])
		
	for i in range(0,M):
		line = input()
		l = line.split()
		ls = list(map(int,l))
		G[ls[0]].append(ls[1])
		GR[ls[1]].append(ls[0])
	return (G,GR) # returns the graph and its reverse
			
def kosaraju():

	(G,GR) = readGraph()
	N = len(G)
	Marked = [0]*N	  # mark vertices
	ExitOrdered = []  # stack to store the vertices ordered by exit no
	Components = [] # A list of sets, one per component : the output
	CompNo = 0 # component number

	# DFS to compute exit number ordering
	def exNoDFS(v):
		nonlocal G,N,ExitOrdered,Marked
		Marked[v] = 1
		for w in G[v]:
			if Marked[w] == 0: 
				exNoDFS(w)
		ExitOrdered.append(v)  # push on stack when you exit
	
	# DFS to output components

	def sccDFS(v): # routine DFS -- adds vertices to current component
		nonlocal GR,N,ExitOrdered,CompNo,Marked,Components
		Components[CompNo] = Components[CompNo] | {v}
		Marked[v] = 1
		for w in GR[v]:
			if Marked[w] == 0:
				sccDFS(w)

	for i in range(0,N):   # generate list ordered by exit number
		if Marked[i] == 0 : 
			exNoDFS(i)  
	Marked = [0]*N   # reset Marked list
	while not (ExitOrdered == []):  
		i = ExitOrdered.pop()
		if Marked[i] == 0:   # new component
			Components.append(set())
			CompNo = len(Components) - 1
			sccDFS(i)

	for s in Components:
		print(s)
		

kosaraju()
