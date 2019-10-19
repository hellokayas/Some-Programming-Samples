from DisjSet import DisjSet

# Graph vertices are number 0 .. N-1 and M is the number of edges
def readEdgeList():
	line = input()
	l = line.split()
	N = int(l[0])
	M = int(l[1])
	G = [] #  List of the form (wt,vertex1,vertex2)
	for i in range(0,M):
		line = input()
		l = line.split()
		ls = list(map(int,l))
		G.append((ls[2],ls[0],ls[1])) # put weight first
	return (N,G)

def mcst(): 
	(N,G) = readEdgeList()
	G.sort(reverse=True)
	S = DisjSet(N)
	Tree = []
	Wt = 0
	while G != [] :
		(w,u,v) = G.pop()
		su = S.find(u)
		sv = S.find(v)
		if su != sv:
			S.merge(su,sv)
			Tree.append((u,v,w))
			Wt = Wt + w
	print(Wt)
	print(Tree)

mcst()
