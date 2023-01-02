S = int(input())
total, index = 0, 1
while S >= total :
    total += index
    index += 1

print(index - 2)