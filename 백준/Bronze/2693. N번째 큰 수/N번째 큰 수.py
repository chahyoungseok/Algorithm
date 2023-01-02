import sys

T = int(input())
for _ in range(T) :
    A = list(map(int, (sys.stdin.readline()).split()))
    max_values = []
    for _ in range(3) :
        value = 0
        for i in A:
            if i > value and i not in max_values:
                value = i
        max_values.append(value)
    print(max_values[2])