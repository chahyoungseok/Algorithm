import sys

n = int(sys.stdin.readline().strip())
A, B, C, D = [], [], [], []
for _ in range(n) :
    a, b, c, d = map(int, (sys.stdin.readline()).split())
    A.append(a)
    B.append(b)
    C.append(c)
    D.append(d)

AB = {}
for a in A :
    for b in B :
        AB[a + b] = AB.get(a + b, 0) + 1

count = 0
for c in C :
    for d in D :
        count += AB.get(-(c + d), 0)

print(count)