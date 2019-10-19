def subsequences(L,n):
     return [L[i:i+n] for i in range(len(L)-n+1)]
def sumallsubsqn(L):
     New = []
     for n in range(1,len(L)+1):
          N = (subsequences(L,n))
          for i in N:
               New.append(i)
     C = []
     for i in New:
          C.append(sum(i))
     return C
     
def indexsum(L,R,Ar):
     T = sumallsubsqn(Ar)
     return sum(T[L:R+1])

T = int(input())
for i in range(0,T):
     line = input()
     l = line.split()
     y = list(map(int,l))
     for i in range(0,y[0]):
          line1 = input()
     l1 = line1.split()
     hello = list(map(int,l1))
     ls = []
     for j in range(0,y[1]):
          line2 = input()
          l2 = line2.split()
          l3 = list(map(int,l2))
          ls.append(l3)

newarray = sumallsubsqn(hello)
for i in range(0,len(ls)):
     print("case #",i,":",indexsum(ls[i][0],ls[i][1],newarray))
