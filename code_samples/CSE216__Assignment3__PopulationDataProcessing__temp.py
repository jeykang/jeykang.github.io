import random
def temp(n):
    if random.choice(range(30)) > 0:
        return "0"+temp(n-1)+"11"
    else:
        return ""

print(temp(10))