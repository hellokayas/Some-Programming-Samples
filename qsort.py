def qsort(l,b,e):
    if b>=e:
        return
    pivot = l[b]
    p=b
    q=e
    while (p<=q):
        while(p<=e and l[p]<=pivot):
            p = p + 1
        while (q>=b and l[q]>pivot):
            q = q-1
        if p<q:
            l[p],l[q] = l[q],l[p]
            p = p+1
            q = q-1
    l[b],l[q] = l[q],l[b]
    qsort (l,b,q-1)
    qsort (l,q+1,e)

def quicksort(l):
    ls = l
    qsort(ls,0,(len(ls)-1))
    return (ls)
M=[1,3,2,5,0,0,1]
quicksort(M)
print(M)
