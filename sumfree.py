def sublist(n):
     array = []


     def sub(k):
          def f(ls):
               xs=ls+[k]
               return xs
          if  k == 1:
               array.append([[],[1]])
               return ([[],[1]])
          x = [[]]
          for j in range(0,k-1):
               x = x + list(map(f,array[j]))
          array.append(x)
          return x
                 
     ans = []
     for i in range(1,n+1):
          ans = ans + sub(i)
     return ans
    
def nub(xs):
         m=[]
         for i in xs:
              if i not in m:
                   m.append(i)
         return m

def allsum(xs):
     m = []
     for i in xs:
          for j in xs:
               if j<i:
                    m.append(j+i)
                    j = j +1
          i = i + 1
     return m

def sumfree(n):
     if n == 0:
          return [[]]
     m = []
     t = nub(sublist(n))
     for i in t:
          tmp=1
          for j in range(0,len(i)-1):
               tmp=1
               for k in range(j+1,len(i)):
                    tmp=1
                    if (i[j]+i[k]) in (i):
                         tmp=0
                    if tmp==0 :break
               if tmp==0 :break
          if tmp==1 :
               m.append(i)          
     return nub(m)



































