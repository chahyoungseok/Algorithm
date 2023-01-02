from collections import deque


def solution(n, t, m, timetable):
    timetable = sorted(timetable)
    standard, timetable = [], deque(timetable)
    for i in range(n):
        standard.append(540 + (i * t))

    def confrim_zero(clock) :
        hour, minute = clock.split(":")
        if len(hour) == 1 :
            hour = "0" + hour
        if len(minute) == 1:
            minute = "0" + minute

        return hour + ":" + minute

    def strtoInt(clock):
        hour, minute = clock.split(":")
        return int(hour) * 60 + int(minute)

    def InttoStr(clock):
        hour, minute = clock // 60, clock % 60
        return str(hour) + ":" + str(minute)

    for i in range(n - 1):
        count = 0
        while m > count and standard[i] >= strtoInt(timetable[0]):
            timetable.popleft()
            count += 1

    if len(timetable) >= m:
        target = strtoInt(timetable[m - 1]) - 1
        if target > standard[n - 1] :
            return confrim_zero(InttoStr(standard[n - 1]))
        else :
            return confrim_zero(InttoStr(target))
    else:
        return confrim_zero(InttoStr(standard[n - 1]))