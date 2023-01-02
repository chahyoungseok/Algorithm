from collections import deque


def solution(n, results):
    answer = 0
    losser_arr = [[] for _ in range(n + 1)]
    winner_arr = [[] for _ in range(n + 1)]

    for result in results :
        winner, losser = result
        winner_arr[losser].append(winner)
        losser_arr[winner].append(losser)

    for i in range(1, n + 1) :
        count, winner_q, losser_q = 0, deque(), deque()
        visited = [False] * (n + 1)

        winner_q.append(i)
        while winner_q :
            node = winner_q.popleft()

            for next_node in winner_arr[node] :
                if not visited[next_node] :
                    winner_q.append(next_node)
                    visited[next_node] = True
                    count += 1

        visited = [False] * (n + 1)

        losser_q.append(i)
        while losser_q:
            node = losser_q.popleft()

            for next_node in losser_arr[node]:
                if not visited[next_node]:
                    losser_q.append(next_node)
                    visited[next_node] = True
                    count += 1

        if count == n - 1 :
            answer += 1

    return answer