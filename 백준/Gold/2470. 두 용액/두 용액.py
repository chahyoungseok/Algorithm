N = int(input())
N_list = list(map(int, input().split()))

A = []
for i in N_list :
    if i > 0 :
        A.append([i, 0])
    else :
        A.append([-i, 1])

A = sorted(A, key=lambda x : x[0])

min_sel, s, e = int(1e11), -1, -1
for i in range(N - 1) :
    if A[i + 1][1] != A[i][1] :
        cost = A[i + 1][0] - A[i][0]
    else :
        cost = A[i + 1][0] + A[i][0]

    if min_sel > abs(cost) :
        min_sel, s, e = abs(cost), i, i + 1

result = []

if A[s][1] == 1 :
    result.append(-A[s][0])
else :
    result.append(A[s][0])

if A[e][1] == 1 :
    result.append(-A[e][0])
else:
    result.append(A[e][0])

result = sorted(result)
print(str(result[0]) + " " + str(result[1]))
