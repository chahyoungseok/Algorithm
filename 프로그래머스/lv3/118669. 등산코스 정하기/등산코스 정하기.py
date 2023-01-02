import heapq


def solution(n, paths, gates, summits):
    edges = [[] for _ in range(n + 1)]
    for i, j, w in paths:
        edges[i].append([j, w])
        edges[j].append([i, w])

    gates.sort()
    summits.sort()
    summits_set = set(summits)

    q, min_dist, min_summit = [], int(1e9), int(1e9)
    for gate in gates:
        heapq.heappush(q, [0, gate])

    distances = [int(1e9) for _ in range(n + 1)]

    while q:
        dist, node = heapq.heappop(q)

        if dist > distances[node] :
            continue

        if dist > min_dist :
            break

        for i in edges[node]:
            cost = max(dist, i[1])
            if i[0] in gates:
                continue
            if distances[i[0]] > cost:
                distances[i[0]] = cost
                if i[0] not in summits_set :
                    heapq.heappush(q, [cost, i[0]])
                elif min_dist > cost:
                    min_dist, min_summit = cost, i[0]
                elif min_dist == cost:
                    min_summit = min(min_summit, i[0])

    return [min_summit, min_dist]