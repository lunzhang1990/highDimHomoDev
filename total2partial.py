def binsort(l):
    prev =-1 
    current = 0
    while current < len(l):
        
        if l[current] >=0:
            current+=1
        else:
            if current -prev > 1:
                l[prev+1:current] = sorted(l[prev+1:current])
            
            prev = current
            current+=1
    if current - prev >1:
        l[prev+1:current] = sorted(l[prev+1:current])
    return tuple(l)
#initial bound = len(path)
#the number of total partial ordering is (n+m)!/n!m!
#res is a set

def total2partial(m,lbound,path,res):
    if m==0:
        res.add(tuple(binsort(path.copy())))
        return
    for i in range(lbound,len(path)+1):
        path[i:i] = [-m]
        total2partial(m-1,i+1,path,res)
        path[i:i+1] = []
        
# check total is a subset of partial
# linear complexity for the length of total and partial
def totalInducePartial(total,partial):
    order = dict()
    for i in range(len(total)):
        order[total[i]]=i
    premax =currentmax=-float('Inf')
    currentmin = float('Inf')
  
    for i in range(len(partial)):
        if partial[i]>=0:
            currentmax = max(currentmax, order[partial[i]])
            currentmin = min(currentmin,order[partial[i]])
        else: # need to go to the next bin
            if currentmin < premax:
                return False
            else:
                premax = currentmax
                currentmin = float('Inf')
                currentmax = -float('Inf')
    if currentmin!=float('Inf') and currentmin < premax:
        return False
    return True  
        
def getPartialOrderByThreshold(m,file):
    res = set()
    with open(file,'r') as file:
        l = file.readline()
        while l:
            l = list(map(lambda x: int(x), l[:-1].split(' ')))
            total2partial(m,0,l,res)
            l = file.readline()
    return res