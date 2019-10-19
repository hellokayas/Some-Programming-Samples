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
'''def dijkstra(G2,s):

     T = len(G2)
     infty = sum(list(map(lSum,G2))) + 1

     Marked = []
     distHeap = Heap(mycmp,[])
     print(T)
     print(list(range(0,T)))
     rit = 0
     while (rit<T):
          print("sayak")
          Marked.append(infty)
          distHeap.insert((i,infty))
          distHeap.insert((s,0))
          rit = rit+1

          while not distHeap.isempty():
               (v,d) = distHeap.deletemin()
               if Marked[v] < infty:
                    continue
               Marked[v] = d
               for e in G2[v]:
                    (u,w) = e
                    print(u,Marked)
                    if Marked[u] == infty :
                         distHeap.insert((u,d+w))
          return Marked'''
