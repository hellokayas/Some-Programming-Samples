from Queue import Queue
#returns the reachable vertices, assuming G is represented via adjacency lists,
#along with the distance from the source
def reachable(G,v):
	rset = {}
	rset[v] = 0 # w is marked (in the keys of rset), with distance
	myq = Queue([v])
	while (not myq.isempty()):
		w = myq.dequeue()
		for ngh in G[w]:
			if not (ngh in rset.keys()):
				rset[ngh] = rset[w]+1
				myq.enqueue(ngh)
		
	return(rset)

G = {}
G[0] = {1,2,3}
G[1] = {0,3}
G[2] = {}
G[3] = {1,4,2}
G[4] = {5,6}
G[5] = {4}
G[7] = {5,6}
G[6] = {5,7}

print(reachable(G,0))
print(reachable(G,1))
print(reachable(G,2))
print(reachable(G,3))
print(reachable(G,4))
print(reachable(G,5))
print(reachable(G,6))
print(reachable(G,7))
