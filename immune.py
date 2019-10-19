def divisible(n, lst):
  return not any(map(lambda y: n%y == 0, lst))

def numtol(n):
	return list(map(int,str(n)))

def isoddl(S):
	z = 0
	for i in S:
		if i % 2 == 0:
			z = 1
	if z == 0:
		return True


T = int(input())
A = []
for i in range(0,T):
     line = input()
     l = line.split()
     y = list(map(int,l))
     A.append(y)

for j in range(0,T):
	M = A[j]

	Q = []
	for i in range(M[0],M[1]+1):
		if ((isoddl(numtol(i)) == True) and (divisible(i,numtol(i)) == True)):
			Q.append(i)

	if len(Q) < M[2]:
		print("-1")
	else:
		print(Q[M[2]-1])
	

