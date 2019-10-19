line = input()
l = line.split()
N = int(l[0])
M = int(l[1])
G = []
num = 0
order = [0]*N
edge = []
rset = set()
parent = [0]*N
def dfs(w):
     global num
     global order
     rset.add(w)
     
  
          
     for ngh in G[w]:
          if not (ngh in rset):
               parent[ngh] = w
               num = num + 1
               order[ngh] = num
               edge.append((ngh,w,1))
               rset.add(ngh)
               dfs(ngh)
          else:
               if parent[w] != ngh and ngh != 0:
                    edge.append((ngh,w,order[ngh] - order[w]))
                    
               
for i in range(0,N):
     G.append([])
		
for i in range(0,M):
     line = input()
     l = line.split()
     ls = list(map(int,l))
     G[ls[0]].append(ls[1])
     G[ls[1]].append(ls[0])
dfs(0)
for E in edge:
          (x,y,z) = E
          print(x,y,z)
          




          

          
