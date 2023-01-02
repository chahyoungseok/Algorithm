def dfs(number) :
    if visited[number] :
        return False
    visited[number] = True

    for i in cow[number] :
        if matched[i] == 0 or dfs(matched[i]):
            matched[i] = number
            return True
    return False


N, M = map(int, input().split())

cow, matched = [[] for _ in range(N + 1)], [0 for _ in range(M + 1)]
for i in range(1, N + 1) :
    data = list(map(int, input().split()))
    for j in range(1, data[0] + 1) :
        cow[i].append(data[j])

for i in range(1, N + 1) :
    visited = [False] * (N + 1)
    dfs(i)

total = 0
for i in range(1, M + 1) :
    if matched[i] != 0 :
        total += 1

print(total)