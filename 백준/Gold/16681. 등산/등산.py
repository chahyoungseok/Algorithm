import sys, heapq


# 해당 문제는 Dijkstra 내부에서 성취감과 거리를 모두 계산한 "가중치" 를 기준으로 삼아 계산을 한다면 문제가 발생한다.
# 이유는 거리대비 큰 성취감의 간선을 마주하기 전 가중치 비교에 의해 걸러질 수 있기 때문이다.
# 따라서 다익스트라 알고리즘 사용시 거리를 기준으로 하지 않을 시 이런 문제가 발생할 우려가 있는지 확인해보자.
def dijkstra(start_node):
    downhill_distances = [INF for _ in range(N + 1)]
    downhill_distances[start_node] = 0
    downhill_q = [[0, start_node]]

    while downhill_q:
        dist, node = heapq.heappop(downhill_q)

        if dist > downhill_distances[node]:
            continue

        for edge in edges[node]:
            current_height, next_height = h_list[node], h_list[edge[0]]
            if next_height > current_height:

                next_dist = dist + edge[1]
                if downhill_distances[edge[0]] > next_dist:
                    downhill_distances[edge[0]] = next_dist
                    heapq.heappush(downhill_q, [next_dist, edge[0]])

    return downhill_distances


N, M, D, E = map(int, sys.stdin.readline().strip().split())

INF = sys.maxsize
h_list = [0] + list(map(int, sys.stdin.readline().strip().split()))

edges = [[] for _ in range(N + 1)]
for _ in range(M):
    a, b, n = map(int, sys.stdin.readline().strip().split())
    edges[a].append([b, n])
    edges[b].append([a, n])

up_distances, down_distances = dijkstra(1), dijkstra(N)
max_value = -INF
for i in range(2, N):
    value = (E * h_list[i]) - ((up_distances[i] + down_distances[i]) * D)
    if value > max_value:
        max_value = value


if max_value == -INF:
    print("Impossible")
else:
    print(max_value)
