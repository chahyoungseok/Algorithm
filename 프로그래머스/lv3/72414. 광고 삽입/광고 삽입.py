def InttoForm_Clock(clock) :
    hour = clock // 3600
    clock = clock % 3600
    minute = clock // 60
    clock = clock % 60

    if hour < 10 :
        hour = "0" + str(hour)

    if minute < 10 :
        minute = "0" + str(minute)

    if clock < 10 :
        clock = "0" + str(clock)

    return str(hour) + ":" + str(minute) + ":" + str(clock)


def ArrtoInt_Clock(arr) :
    return int(arr[0]) * 3600 + int(arr[1]) * 60 + int(arr[2])


def solution(play_time, adv_time, logs):
    total_clock, current_index = 0, 0
    play_time, adv_time = ArrtoInt_Clock(play_time.split(":")), ArrtoInt_Clock(adv_time.split(":"))
    dynamic = [0 for _ in range(play_time + 1)]

    if play_time == adv_time :
        return "00:00:00"

    for log in logs :
        start, end = log.split("-")
        dynamic[ArrtoInt_Clock(start.split(":"))] += 1
        dynamic[ArrtoInt_Clock(end.split(":"))] -= 1

    end_index = 0
    for i in range(adv_time) :
        end_index += dynamic[i]
        total_clock += end_index
        max_time = total_clock

    start_index = 0
    for i in range(1, play_time - adv_time + 1) :
        start_index += dynamic[i - 1]
        end_index += dynamic[i + adv_time - 1]
        total_clock = total_clock - start_index + end_index
        if total_clock > max_time :
            total_clock = max_time
            current_index = i
    return InttoForm_Clock(current_index)