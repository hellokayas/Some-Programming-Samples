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
#print(G)

'''G1 = []
N = len(G)
for i in range(0,2*N):
     G1.append([])
for i in range(N):
     for j in G[i]:
          (x,y) = j
          G1[i].append((x+N,y))
          G1[i+N].append((x,y))
'''
G1=[]
N=len(G)
for i in range(N):
     l=[]
     for j in G[i]:
          v,w=j
          l.append((v+N,w))
     G1.append(l)
for i in G:
     G1.append(i)
#print(G1)

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
          distHeap.insert((i,infty))
     distHeap.insert((s,0))
     while not distHeap.isempty():
          (v,d) = distHeap.deletemin()
     #print("here")
      #    print(distHeap.heap)
       #   print((v,d))
        #  print(Marked)
          if Marked[v] < infty:
               continue
          Marked[v] = d
          for e in G[v]:
            #   print("e")
             #  print(e)
               (u,w) = e
               if Marked[u] == infty :
                    distHeap.insert((u,d+w))
     return Marked

     
s = int(input())
Q=dijkstra(G1,s)
for i in range(0,N):
     print(Q[i],end=" ")

