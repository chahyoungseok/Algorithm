N = int(input())

time = [0] * N
prerequisites = [[] for _ in range(N)]

for i in range(N) :
    data = list(map(int, input().split()))
    time[i] = data[0]
    prerequisites[i] = data[1:-1]

subjecting, time_sum = [], 0
for i in range(N) :
    if not prerequisites[i] :
        subjecting.append([time[i], i])

while subjecting :
    min_time = int(1e9)
    for i in subjecting :
        if min_time > i[0] :
            min_time = i[0]
    time_sum += min_time

    for i in range(len(subjecting)) :
        subjecting[i][0] -= min_time
        if subjecting[i][0] == 0 :
            for j in range(N) :
                if (subjecting[i][1] + 1) in prerequisites[j] :
                    prerequisites[j].remove(subjecting[i][1] + 1)
                    if not prerequisites[j] :
                        subjecting.append([time[j], j])
            time[subjecting[i][1]] = time_sum
            subjecting.remove(subjecting[i])

print(time)