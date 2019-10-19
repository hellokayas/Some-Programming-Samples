
line = input()
l = line.split()
M = int(l[0])#rows
N = int(l[1])#col
mat = []
for i in range(0,M):
     line = input()
     line2 = list(map(int,line.split()))
     mat.append(line2)
adj = []# each list inside gives row and each singleton list inside some list gives the col elem
for i in range(0,M):
     hello = []
     for j in range(0,N):
          hello.append([])
     adj.append(hello)
        
rset = []
rset1 = []
def dfs(i,j):
     global rset
     if (i,j) in rset:
          return
     rset.append((i,j))
     if (i+1) in range(0,M) and j in range(0,N):
          if mat[i][j]>mat[i+1][j]:
               adj[i][j].append((i+1,j))
               if not ((i+1,j) in rset):
                    dfs(i+1,j)
                    
     if i in range(0,M) and (j+1) in range(0,N):
          if mat[i][j]>mat[i][j+1]:
               adj[i][j].append((i,j+1))
               if not ((i,j+1) in rset):
                    dfs(i,j+1)
                    
     if (i-1) in range(0,M) and j in range(0,N):
          if mat[i][j]>mat[i-1][j]:
               adj[i][j].append((i-1,j))
               if not ((i-1,j) in rset):
                    dfs(i-1,j)
                    
     if i in range(0,M) and (j-1) in range(0,N):
          if mat[i][j]>mat[i][j-1]:
               adj[i][j].append((i,j-1))
               if not ((i,j-1) in rset):
                    dfs(i,j-1)
                    
     rset1.append((i,j))
     return
     
				
length = []
for i in range(0,M):
     say = [-1]*N
     length.append(say)
rit = []
for i in range(0,M):
     say= [(-1,-1)]*N
     rit.append(say)

def mymax(i,j):
     global length,adj
     (r,s) = adj[i][j][0]
     l = length[r][s]
     for k in range(1,len(adj[i][j])):
          (t,u) = adj[i][j][k]
          if length[t][u]>l:
               l = length[t][u]
               (r,s)= (t,u)
     return((l,r,s))

def maxi():
     global length
     m = length[0][0]
     (r,s) = (0,0)
     for i in range(0,M):
          for j in range(0,N):
               if length[i][j]>m:
                    m = length[i][j]
                    (r,s) = (i,j)
     return((m,r,s))

def sequence(i,j):
     print(i+1,j+1)
     if rit[i][j] != (-1,-1):
          (r,s) = rit[i][j]
          sequence(r,s)
     


for i in range(0,M):
     for j in range(0,M):
          dfs(i,j)
#rset.reverse()
print(rset1)
for v in rset1:
     (i,j) = v
     if adj[i][j] == []:
          length[i][j] = 0
     else:
          (x,y,z) = mymax(i,j)
          length[i][j] = 1 + x
          rit[i][j] = (y,z)
(p,q,r) = maxi()
print(p)
sequence(q,r)

"the complexity of the program is O(MN) where M,N are the no. of rows and columns resp"          

