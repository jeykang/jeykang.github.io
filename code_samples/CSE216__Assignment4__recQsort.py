def partition(num, low, high):
    pivot = num[high]
    start = low - 1
    for i in range(low, high):
        if num[i] <= pivot:
            start += 1
            temp = num[i]
            num[i] = num[start]
            num[start] = temp
    temp = num[start + 1]
    num[start + 1] = num[high]
    num[high] = temp
    return start + 1

def quickSort(num, low, high):
    if low < high:
        pi = partition(num, low, high)
        quickSort(num, low, pi - 1)
        quickSort(num, pi + 1, high)

num = [26, 4, 90, 14, 52, 11, 95, 83, 35, 31, 98, 30, 81, 32, 67] 
quickSort(num, 0, len(num) - 1)
print(num)
#temp = [1, 2, 5, 1]
#print(partition(num, 0 , len(num)-1), num)