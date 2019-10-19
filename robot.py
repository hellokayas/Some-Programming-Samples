A = []
T = int(input())
for i in range(0,T):
     line = input()
     l = line.split()
     y = list(map(int,l))
     up = 0
     down = 0
     right = 0
     left = 0
     a = y[3] - y[1]
     b = y[2] - y[0]
     if a>0:
     	right = 1
     if a<0:
     	left = 1
     if b>0:
     	up = 1
     if b<0:
     	down = 1
     if (right == 1 and left == 0 and up == 0 and down == 0):
     	A.append("right")
     if (right == 0 and left == 1 and up == 0 and down == 0):
     	A.append("left")
     if (right == 0 and left == 0 and up == 1 and down == 0):
     	A.append("up")
     if (right == 0 and left == 0 and up == 0 and down == 1):
     	A.append("down")
     if (right == 1 and left == 0 and up == 1 and down == 0):
     	A.append("sad")
     if (right == 1 and left == 0 and up == 0 and down == 1):
     	A.append("sad")
     if (right == 0 and left == 1 and up == 1 and down == 0):
     	A.append("sad")
     if (right == 0 and left == 1 and up == 0 and down == 1):
     	A.append("sad")

for i in range(0,T):
	print(A[i],"\n")