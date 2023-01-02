from collections import deque


def check_distance(arr, colum, row):
    queue = deque()
    queue.append([colum, row, 0])
    visited = [[False] * 5 for _ in range(5)]
    while queue:
        col, ro, dist = queue.popleft()
        visited[col][ro] = True

        if col - 1 >= 0 and not visited[col - 1][ro] and arr[col - 1][ro] != 'X':
            if arr[col - 1][ro] == 'P':
                return False
            if dist < 1 :
                queue.append([col - 1, ro, dist + 1])
        if ro - 1 >= 0 and not visited[col][ro - 1] and arr[col][ro - 1] != 'X':
            if arr[col][ro - 1] == 'P':
                return False
            if dist < 1 :
                queue.append([col, ro - 1, dist + 1])
        if col + 1 < 5 and not visited[col + 1][ro] and arr[col + 1][ro] != 'X':
            if arr[col + 1][ro] == 'P':
                return False
            if dist < 1:
                queue.append([col + 1, ro, dist + 1])
        if ro + 1 < 5 and not visited[col][ro + 1] and arr[col][ro + 1] != 'X':
            if arr[col][ro + 1] == 'P':
                return False
            if dist < 1:
                queue.append([col, ro + 1, dist + 1])

    return True


def solution(places):
    answer = []
    for place in places:
        state = True
        for colum in range(5):
            for row in range(5):
                if place[colum][row] == 'P':
                    state = check_distance(place, colum, row)
                if not state :
                    break
            if not state :
                break
        answer.append(int(state))
    return answer