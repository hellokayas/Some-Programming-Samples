def qsort(l,b,e):
    if b>=e:
        return
    p = b
    q = p+1
    while (q<=e):
        if (l[q] > l[b]):
            q=q+1
        else:
            l[q],l[p+1] = l[p+1],l[q]
            p = p+1
            q=  q+1
    l[b],l[p] = l[p],l[b]
    qsort(l,b,p-1)
    qsort(l,p+1,e)

def quicksort(l):
    ls = l
    qsort(ls,0,len(ls)-1)
    return (ls)
M=[2,3,5,8,7,4,5,4,1,2,5,4,8,9,6,5,4,7,8,5,4,1,25,4,7,8]
quicksort(M)
print(M)
