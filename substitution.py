import cmath
import string


def subencry(n,m):
     y = m.split()
     l = "abcdefghijklmnopqrstuvwxyz"
     if n not in range(0,25):
          print("this is not a valid value for the shift")
     z = []
     for i in range(0,25):
          z.append(l[((i+n)%26)])
     t = ""
     for j in y:
          for i in j:
               k = l.index(i)
               t = t + str(z[k])
     return t

'''def subdcry(m):
     s = []
     q= []
     for i in range(0,25):
          s.append(subencry(i,m))
          print(s)
     flag = 0
     for j in s:
          if ('a' not in j) and ('e' not in j) and ('i' not in j) and ('o' not in j) and  ('u' not in j):
               flag = 1
          else:
               flag = 0
          if flag == 0:
               print(j)
               q.append(j)
     for i in q:
          if detect(i) == 'en':
               print(i)'''
               
        
def decrot(m):
     s = subencry(13,m)
     print(s)
''' i was trying to give a generalized decryption pattern for any substitution cipher. for the encryption i have succeeded but i am not being able to do the decryption part.
The part commented gives all possible decryption patterns of the cipher. i can't understand how to teach the computer to detect english language such that it gives the exact correct message'''
