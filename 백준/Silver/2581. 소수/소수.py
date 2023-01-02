M = int(input())
N = int(input())
if M == 1 :
    M = 2

N += 1
state = [True] * N

for i in range(2, int(N ** 0.5) + 1) :
    if state[i] :
        for j in range(i + i, N, i) :
            state[j] = False

primeList = [i for i in range(M, N) if state[i]]

if primeList :
    print(sum(primeList))
    print(primeList[0])
else :
    print(-1)