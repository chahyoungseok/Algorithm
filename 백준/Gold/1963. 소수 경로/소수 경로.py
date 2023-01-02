import copy, sys
from collections import deque

state = [True for _ in range(10000)]

for i in range(2, int(10000 ** 0.5) + 1):
    if state[i]:
        for j in range(i + i, 10000, i):
            state[j] = False

primeNumber = {str(i) : True for i in range(1000, 10000) if state[i]}
prime = list(primeNumber.keys())

T = int(sys.stdin.readline())
for _ in range(T) :
    A, B = map(int, (sys.stdin.readline()).split())

    visited = copy.deepcopy(primeNumber)
    visited[str(A)], max_distance = False, 0
    q, A, B = deque(), str(A), str(B)
    q.append([str(A), 1])
    state = False

    while q :
        current, dist = q.popleft()

        for k in range(4) :
            for n in range(10) :
                trans = current[:k] + str(n) + current[k + 1:]
                if trans == B :
                    max_distance, state = dist, True
                    break

                if trans in prime and visited[trans] :
                    visited[trans] = False
                    q.append([trans, dist + 1])

            if state :
                break
        if state :
            break
            
    if not state:
        print("Impossible")
    elif A == B :
        print(0)
    else :
        print(max_distance)
