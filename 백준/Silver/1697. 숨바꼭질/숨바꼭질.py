from collections import deque

N, K = map(int, input().split())

if N >= K :
    print(abs(K - N))
else :
    q, result = deque(), []
    q.append([N, 0])

    dp = [0] * 100001
    dp[N] = 0

    for i in range(N - 1, -1, -1):
        dp[i] = dp[i + 1] + 1

    for i in range(N, 100000, 1):
        dp[i + 1] = dp[i] + 1

    while q :
        current, dist = q.popleft()
        if current == K :
            result.append(dist)
            continue

        if dist > dp[current] :
            continue

        dp[current] = min(dp[current], dist)

        if (current + 1) >= 0 and (current + 1) < 100001 :
            q.append([current + 1, dist + 1])
        if (current - 1) >= 0 and (current - 1) < 100001:
            q.append([current - 1, dist + 1])
        if (current * 2) >= 0 and (current * 2) < 100001:
            q.append([current * 2, dist + 1])
    print(min(result))