n = int(input())
measures = [2,3,5]
ugly_number = [1]
index = 0

while len(ugly_number) <= n :
    for measure in measures :
        if not ugly_number[index] * measure in ugly_number :
            ugly_number.append(ugly_number[index] * measure)
    index += 1

ugly_number.sort()
print(ugly_number[n - 1])