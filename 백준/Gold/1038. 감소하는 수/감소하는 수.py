# 38
from collections import deque

N = int(input())
data = []
q = deque(i for i in range(10))

while q :
    current = q.popleft()
    data.append(current)

    for i in range(int(str(current)[0]) + 1, 10) :
        q.append(int(str(i) + str(current)))

data = sorted(data)
if N >= len(data) :
    print(-1)
else :
    print(data[N])