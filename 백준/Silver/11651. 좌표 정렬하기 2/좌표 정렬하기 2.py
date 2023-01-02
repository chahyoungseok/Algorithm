import sys

N = int(sys.stdin.readline().strip())
arr = []
for _ in range(N) :
    arr.append(list(map(int, (sys.stdin.readline().split()))))

arr = sorted(arr, key=lambda x : (x[1], +x[0]))
for a, b in arr :
    print(str(a) + " " + str(b))