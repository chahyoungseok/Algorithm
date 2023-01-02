import sys
from collections import deque

N = int(input())
K = int(input())
apples = []
for _ in range(K) :
    apples.append(list(map(int, (sys.stdin.readline()).split())))

L = int(input())
dx, dy = [0, 1, 0, -1], [1, 0, -1, 0]
move, data, current_location, current_time = deque(), [], 0, 0
cx, cy, snack_body = 1, 1, deque()
snack_body.append([1, 1])
for _ in range(L) :
    data.append(list(map(str, sys.stdin.readline().split())))

move.append([int(data[0][0]), "N"])
for i in range(L - 1) :
    move.append([int(data[i + 1][0]) - int(data[i][0]), data[i][1]])
move.append([101, data[L - 1][1]])

state = True
while move and state:
    second, location = move.popleft()
    if location == "D" :
        current_location = (current_location + 1) % 4
    elif location == "L" :
        current_location = (current_location + 3) % 4

    for _ in range(int(second)) :
        mx, my = cx + dx[current_location], cy + dy[current_location]
        current_time += 1
        if 1 <= mx < N + 1 and 1 <= my < N + 1 and [mx, my] not in snack_body :
            if [mx, my] in apples :
                apples.remove([mx, my])
            else:
                snack_body.popleft()
            snack_body.append([mx, my])
            cx, cy = mx, my
        else :
            state = False
            break

print(current_time)