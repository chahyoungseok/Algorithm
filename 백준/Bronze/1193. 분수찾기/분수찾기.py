import sys

N = int(sys.stdin.readline().strip())

index = 1
while N > 0 :
    N -= index

    index += 1
result = index + N - 1
if index % 2 == 1 :
    print(str(result) + "/" + str(index - result))
else :
    print(str(index - result) + "/" + str(result))