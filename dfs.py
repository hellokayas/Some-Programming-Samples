#returns the set of reachable vertices assuming G is represented via adjacency lists.
# Assumes that the graph is a dictionary and each adjacency list is a set.

def reachable(G,v):
	rset = set()
	def dfs(w):
		rset.add(w)
		for ngh in G[w]:
			if not (ngh in rset):
				dfs(ngh)
		return
	dfs(v)
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
