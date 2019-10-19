def find_permutations(v):
    if len(v) > 1:
        for i in perms(v):
            nv = i[1:]
            for j in perms(nv):
                print(i[0] + j)
    else:
        print(v)


def perms(v):
    if not hasattr(perms, 'original'):
        perms.original = v
        perms.list = []
    nv = v[1:] + v[0]
    perms.list.append(nv)
    if perms.original == nv:
        l = perms.list
        del perms.original
        del perms.list
        return l
    return perms(nv)
