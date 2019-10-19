import operator

class Heap:

	def __init__(self,fn=operator.le,l = []):
		self.cmpfn = fn
		self.__makeheap(l)

	def par(self,c):
		return((c-1)//2)
	
	def isempty(self):
		return (self.N == 0)
	
	def insert(self,v):
		self.heap.append(v) 
		self.N = self.N+1
		cur = self.N - 1
		while (cur > 0) and self.cmpfn(self.heap[cur],self.heap[self.par(cur)]):
			self.heap[cur],self.heap[self.par(cur)] = self.heap[self.par(cur)],self.heap[cur]
			cur = self.par(cur)
		
	def deletemin(self):
		ans = self.heap[0]
		self.heap[0],self.heap[self.N-1] = self.heap[self.N-1],self.heap[0]
		self.N = self.N - 1	
		self.heap.pop()
		cur = 0
		while (cur < self.N):
			left = 2*cur + 1
			right = 2*cur + 2
			if (left >= self.N):
				break
			elif (right >= self.N):
				pos = left
			else:
				if self.cmpfn(self.heap[left],self.heap[right]):
					pos = left
				else:
					pos = right
			if self.cmpfn(self.heap[pos],self.heap[cur]):
				self.heap[cur],self.heap[pos] = self.heap[pos],self.heap[cur]
			cur = pos
		return(ans)

	def __makeheap(self,l):
		self.N = 0
		self.heap = []
		for i in l:
			self.insert(i)	



def transpose( matrix ):
    if not matrix: return []
    return [ [ row[ i ] for row in matrix ] for i in range( len( matrix[ 0 ] ) ) ]         

line = input()
l = line.split()
M = int(l[0])#rows
N = int(l[1])#col
mat = []
for i in range(0,M):
     line = input()
     line2 = list(map(int,line.split()))
     mat.append(line2)

     
A1 = []
for i in range(len(mat)):
     t = []
     for j in range(len(mat[i]) - 1):
          if (mat[i][j] == 1) and (mat[i][j+1] == 1):
               t.append(((i,j),(i,j+1)))
     A1.append(t)

mat1 = transpose(mat)
A2 = []
for i in range(len(mat1)):
     b = []
     for j in range(len(mat1[i]) - 1):
          if (mat1[i][j] == 1) and (mat1[i][j+1] == 1):
               b.append(((j,i),(j+1,i)))
     A2.append(b)

Q = []
for i in A1:
     for j in i:
          Q.append(j)
for i in A2:
     for j in i:
          Q.append(j)
#now to give the edges
d1 = {}
for i in A1:
     for j in range(0,len(i)-1):
          (x,y) = i[j]
          (w,z) = i[j+1]
          if x in i[j+1] or y in i[j+1]:
               d1[Q.index(i[j])] = d1.get(Q.index(i[j]),[]) + [(Q.index(i[j+1]),1)]
               d1[Q.index(i[j+1])] = d1.get(Q.index(i[j+1]),[]) + [(Q.index(i[j]),1)]


d2 = {}
for i in A2:
     for j in range(0,len(i)-1):
          (x,y) = i[j]
          (w,z) = i[j+1]
          if x in i[j+1] or y in i[j+1]:
               d2[Q.index(i[j])] = d2.get(Q.index(i[j]),[]) + [(Q.index(i[j+1]),1)]
               d2[Q.index(i[j+1])] = d2.get(Q.index(i[j+1]),[]) + [(Q.index(i[j]),1)]

d3 = {}
for i in A1:
     for k in range(0,len(i)):
          for j in A2:
               for t in range(0,len(j)):
                    (x,y) = i[k]
                    (w,z) = j[t]
                    if x in j[t] or y in j[t]:
                         d3[Q.index(i[k])] = d3.get(Q.index(i[k]), []) + [(Q.index(j[t]), 2 * M * N)]
                         d3[Q.index(j[t])] = d3.get(Q.index(j[t]), []) + [(Q.index(i[k]), 2 * M * N)]

H1 = list(d1.keys())
H2 = list(d2.keys())
H3  = list(d3.keys())
C = []
#print(d1)
#print(d2)
#print(d3)
for i in Q:
     C.append([])
for i in H1:
     C[i].extend(d1[i])
for i in H2:
     C[i].extend(d2[i])
for i in H3:
     C[i].extend(d3[i])
#print(C)
def starting(Q):
    ret = []
    for i in Q:
         (x,y) = i
         if x == (0,0) or y == (0,0):
              ret.append(Q.index(i))
    return ret

def ending(Q):
    ret = []
    for i in Q:
         (x,y) = i
         if x == (M-1,N-1) or y == (M-1,N-1):
              ret.append(Q.index(i))
    return ret

'''d = {}
for i in ls:
	(n,m) = i
	d[n] = d.get(n,0) + m'''
###the dijkstra by sir
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
	distHeap = Heap()
	distHeap.cmpfn = mycmp
	for i in range(0,N):
		Marked.append(infty)
	distHeap.insert((s,1))

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


m = []
for i in starting(Q):
    R = dijkstra(C,i)
    for j in ending(Q):
        m.append(R[j])
print(min(m) // (2 * M * N))

