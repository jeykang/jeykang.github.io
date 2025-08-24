def merge(whole, left, right):
    i = 0
    j = 0
    k = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            whole[k] = left[i]
            i += 1
        else:
            whole[k] = right[j]
            j += 1
        k += 1
    while i < len(left):
        whole[k] = left[i]
        i += 1
        k += 1
    while j < len(right):
        whole[k] = right[j]
        j += 1
        k += 1

def mergeSort(num):
    if len(num) > 1:
        mid = len(num) // 2
        left = num[:mid]
        right = num[mid:]
        mergeSort(left)
        mergeSort(right)
        merge(num, left, right)
        
num = [26, 4, 90, 14, 52, 11, 95, 83, 35, 31, 98, 30, 81, 32, 67] 
mergeSort(num)
print(num)