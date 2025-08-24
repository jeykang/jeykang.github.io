from functools import reduce
from math import sqrt

def mean(lst):
    return reduce(lambda x, y: x + y, lst)/len(lst)# if len(lst) > 1 else lst[0]

def median(lst):
    return reduce(lambda x, y: x + y, list(filter(lambda x: x == sorted(lst)[len(lst)//2 - 1] or x == sorted(lst)[len(lst)//2], lst))) / 2

def mode(lst):
    return list(filter(lambda x: lst.count(x) == sorted([lst.count(i) for i in lst])[-1], lst))[0]

def variance(lst):
    return reduce(lambda x, y: x + y, [(i - mean(lst))**2 for i in lst])/(len(lst)-1)# if len(lst) > 1 else 0

def percentchange(lst):
    print(lst)
    return reduce(lambda x, y: (y - x)/x, lst) * 100
    #return (lst[1] - lst[0]) * 100 / lst[0]# if len(lst) > 1 else 0

def meanfromyears(popdict, code, start, end):
    return mean([i[1] for i in list(filter(lambda x: x[0] in range(start, end+1), popdict[code]))])

def medianfromyears(popdict, code, start, end):
    return median([i[1] for i in list(filter(lambda x: x[0] in range(start, end+1), popdict[code]))])

def modefromyears(popdict, code, start, end):
    return mode([i[1] for i in list(filter(lambda x: x[0] in range(start, end+1), popdict[code]))])

def variancefromyears(popdict, code, start, end):
    return variance([i[1] for i in list(filter(lambda x: x[0] in range(start, end+1), popdict[code]))])

def stddevfromyears(popdict, code, start, end):
    return sqrt(variancefromyears(popdict, code, start, end))

def percentchangefromyears(popdict, code, start, end):
    return percentchange([i[1] for i in list(filter(lambda x: x[0] in [start, end], popdict[code]))])
