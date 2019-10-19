from Heap import Heap
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
		G[ls[0]].append((ls[1],ls[2]))
		G[ls[1]].append((ls[0],ls[2]))

	return G
			
def proj(a):
	(x,y) = a
	return y	

def lSum(l):
	return sum(list(map(proj,l)))
	
def mycmp(a,b):
	(x1,y1) = a
	(x2,y2) = b
	return (y1 <= y2)

def dijkstra(G,s):

	N = len(G)
	infty = sum(list(map(lSum,G))) + 1

	Marked = []
	distHeap = Heap(mycmp,[])

	for i in range(0,N):
		Marked.append(infty)
		distHeap.insert((i,infty))
	distHeap.insert((s,0))

	while not distHeap.isempty():
		(v,d) = distHeap.deletemin()
		if Marked[v] < infty:
			continue
		Marked[v] = d
		for e in G[v]:
			(u,w) = e
			if Marked[u] == infty :
				distHeap.insert((u,d+w))
	return Marked

def shortestpath():
	G = readGraph()
	s = int(input())
	print(G)
	print(dijkstra(G,s))
	
shortestpath()	
