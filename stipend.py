def isstip(L):
     l = len(L)
     for i in L:
          if i == 2:
               print ("No")
     if 5 not in L:
          print("No")
     A = (sum(L))/l
     if A >= 4:
          print("Yes")
     else:
          print("No")

T = int(input())
W = []
for i in range(0,T):
     K = []
     N = int(input())
     M = input()
     M1 = M.split()
     for i in range(0,len(M1)):
          t = int(M1[i])
          K.append(t)
          #print(K)
     W.append(K)
     #print(len(W))
for i in range(0,len(W)):
     print(isstip(W[i]))
