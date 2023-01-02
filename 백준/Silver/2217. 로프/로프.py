import sys

N = int(input())
l = []
for _ in range(N) :
    l.append(int(sys.stdin.readline()))

l = sorted(l, reverse=True)

max_weight = 0
for i in range(N) :
    max_weight = max(max_weight, l[i] * (i + 1))

print(max_weight)