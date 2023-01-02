T = int(input())

for _ in range(T) :
    N, M = map(int, input().split())
    result = 1
    for i in range(0, N) :
        result *= (M - i)
        result /= (i + 1)

    print(int(result))