N, M = map(int, input().split())
ball_weights = list(map(int, input().split()))

case_count = 0
weights = [0] * (M + 1)

for x in ball_weights :
    weights[x] += 1

for i in range(1, M + 1) :
    N = N - weights[i]
    case_count += N * weights[i]

print(case_count)