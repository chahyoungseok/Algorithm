import sys

N = int(input())

schedule, min_schedules, real_schedule, overlap = [], [], [], []
for _ in range(N):
    start, end = map(int, (sys.stdin.readline()).split())
    if start == end:
        overlap.append(start)
    else:
        schedule.append([start, end])

schedule = sorted(schedule, key=lambda x : (x[0], x[1]))
for i in schedule:
    if not real_schedule or real_schedule[len(real_schedule) - 1][0] != i[0]:
        real_schedule.append(i)

if not real_schedule:
    print(len(overlap))
else:
    min_schedules.append(real_schedule[0])
    real_schedule.pop(0)
    for i in real_schedule:
        s, e = i
        s_len = len(min_schedules) - 1

        if min_schedules[s_len][1] > s:
            if min_schedules[s_len][1] > e:
                min_schedules.pop(s_len)
            else:
                continue
        min_schedules.append([s, e])

    count = 0
    for i in min_schedules :
        s, e = i
        for j in overlap :
            if e > j and j > s :
                count += 1
                break
    print(len(min_schedules) + len(overlap) - count)
