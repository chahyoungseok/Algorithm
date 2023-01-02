import sys


def mutiple(A, B) :
    A_len = len(A)
    temp = [[0 for _ in range(A_len)] for _ in range(A_len)]

    for i in range(A_len) :
        for j in range(A_len) :
            for k in range(A_len) :
                temp[i][j] += A[i][k] * B[k][j]

    for i in range(A_len) :
        for j in range(A_len) :
            temp[i][j] %= 1000

    return temp


def mod_squared(A, B) :
    if B == 1 :
        for i in range(len(A)):
            for j in range(len(A)):
                A[i][j] %= 1000
        return A

    elif B == 2 :
        return mutiple(A, A)

    if B % 2 == 0 :
        result = mod_squared(A, B // 2)
        return mutiple(result, result)
    else :
        result = mod_squared(A, B // 2)
        return mutiple(mutiple(result, result), A) # B행렬 곱하기기


N, B = map(int, (sys.stdin.readline()).split())
A = []

for _ in range(N) :
    A.append(list(map(int, (sys.stdin.readline()).split())))

result = mod_squared(A, B)

for i in range(len(result)):
    for j in range(len(result)):
        print(result[i][j], end=" ")
    print()