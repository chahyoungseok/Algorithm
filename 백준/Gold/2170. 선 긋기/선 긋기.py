import sys

N = int(sys.stdin.readline().strip())

lines = []
for _ in range(N):
    lines.append(list(map(int, sys.stdin.readline().strip().split())))

lines = sorted(lines, key=lambda x:x[0])

INF = sys.maxsize
current_start = lines[0][0]
current_end = lines[0][1]
if N == 1:
    print(current_end - current_start)
else:
    sum_value = 0
    for idx in range(1, N):
        start, end = lines[idx][0], lines[idx][1]

        if current_end >= end:
            if idx == N - 1:
                sum_value += current_end - current_start
            continue

        if start > current_end:
            sum_value += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = end

        if idx == N - 1:
            sum_value += current_end - current_start

    print(sum_value)
