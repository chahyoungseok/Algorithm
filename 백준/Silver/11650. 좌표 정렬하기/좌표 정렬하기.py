import sys

N = int(input())
location = []
for _ in range(N) :
    location.append(list(map(int, (sys.stdin.readline()).split())))
location = sorted(location, key=lambda x : (x[0], x[1]))
for x, y in location :
    print(str(x) + " " + str(y))