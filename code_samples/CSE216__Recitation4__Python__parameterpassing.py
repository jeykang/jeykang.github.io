def pass_integer(x):
    print ("x=",x," id=",id(x))
    x=42
    print ("x=",x," id=",id(x))

x = 20
print ("x=",x," id=",id(x))
pass_integer(x)
print ("x=",x," id=",id(x))

def pass_list(xlist):
    print (xlist)

xlist = (1,2,3,4,5,6)
pass_list(xlist)
print(xlist)
"""
def change_list(xlist):
    print (xlist)
    xlist += [47, 42]
    print (xlist)

xlist = [1,2,3,4,5,6]
change_list(xlist)
print(xlist)

xlist = [1,2,3,4,5,6]
change_list(xlist.copy())
print(xlist)

def process_list(xtuple):
    print(xtuple)
    xlist = list(xtuple)
    xlist.append((47, 42))
    xtuple=tuple(xlist)
    print(xtuple)

xtuple = (1,2,3,4,5,6)
process_list(xtuple)
print(xtuple)

def var_args(*x):
    print(x)
    print(type(x))

var_args('a','b','c','d','e','f')
"""
