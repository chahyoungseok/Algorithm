import copy, sys

N, K = map(int, (sys.stdin.readline()).split())
tools = list(map(int, (sys.stdin.readline()).split()))

if N >= len(set(tools)):
    print(0)
else :
    plugs, total = [], 0
    for i in range(K) :
        if tools[i] not in plugs :
            plugs.append(tools[i])
        if len(plugs) == N :
            break

    for i in range(N, K) :
        if tools[i] in plugs:
            continue
        else :
            visited = copy.deepcopy(plugs)
            for j in range(i, K) :
                if tools[j] in visited :
                    visited.remove(tools[j])
                if len(visited) == 1 :
                    break
            for j in range(N) :
                if plugs[j] == visited[0] :
                    plugs[j] = tools[i]
                    break
            total += 1
    print(total)