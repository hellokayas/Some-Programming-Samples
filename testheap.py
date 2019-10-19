from Heap import Heap

g = Heap([1,3,2,4,1])
h = Heap()
print(g.N)
while (not g.isempty()):
	print(g.deletemin(),end=" ")

print()
