import sys

N = int(input())
data = []
for _ in range(N) :
    data.append(list(map(str, (sys.stdin.readline()).split())))

data = sorted(data, key=lambda x : int(x[0]))
for y, n in data :
    print(y + " " + n)