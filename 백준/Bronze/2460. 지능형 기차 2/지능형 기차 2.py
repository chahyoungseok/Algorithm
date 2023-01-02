import sys

total = 0
max_total = 0
for _ in range(10) :
    a, b = map(int, (sys.stdin.readline()).split())
    total += (b - a)
    if total > 10000 :
        total = 10000
    max_total = max(max_total, total)

print(max_total)