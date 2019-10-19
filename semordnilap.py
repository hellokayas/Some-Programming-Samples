import string

f = open('program.txt','r')
l = f.read()
m = l.split()
t = []
'''for i in m:
     t.append(''.join(reversed(i)))'''
for i in m:
     if ''.join(reversed(i)) in m:
          print(i,''.join(reversed(i)))
          
