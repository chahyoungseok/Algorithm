import sys


def solution(N, K) :
    count = 0
    states = [True for _ in range(N)]
    for i in range(2, N) :
        if states[i] :
            count += 1
            if count == K :
                return i
            for j in range(i + i, N, i) :
                if states[j] :
                    states[j] = False
                    count += 1
                    if count == K :
                        return j
    return [i for i in range(2, N) if states[i]]


N, K = map(int, (sys.stdin.readline()).split())
print(solution(N + 1, K))
