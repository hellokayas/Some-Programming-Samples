A = "abbcd"
B = "baac"
M = len(A)
N = len(B)

def lcs(i,j):
     if (i>=M) and (j>=N):
          return ("",0)
     (x,y) = lcs(i,j)
     def proj(m,n):
          return n
     def comp(m,n):
          return m
     if (A[i] == B[j]):
          return (x + A[i],1 + lcs(i+1,j+1))
     y1 = proj(lcs(i,j+1)) 
     y2 = proj(lcs(i+1,j)) 
     if y1>=y2:
          return(x+comp(lcs(i,j+1)),y1)
     if y2>y1:
          return(x + comp(lcs(i+1,j)),y2)
	
print(lcs(0,0))
