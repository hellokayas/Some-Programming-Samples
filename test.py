def selection(l,b):
    if (b<len(l)-1):
        #pos = b
        for i in range (b,len(l)):
            if (l[i]<l[b]):
                #pos = i
                l[b],l[i] = l[i],l[b]
        selection(l,b+1)
def sort(l):
     selection(l,0)
M =[4,4,5,5,1,2,3,5,4,0]
sort(M)
print (M)
