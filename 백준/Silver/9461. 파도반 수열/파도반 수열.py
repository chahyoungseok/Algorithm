import sys

T = int(sys.stdin.readline())
tri = [1, 1, 1]
for i in range(3, 101) :
    tri.append(tri[i - 2] + tri[i - 3])

for _ in range(T) :
    N = int(sys.stdin.readline())
    print(tri[N - 1])