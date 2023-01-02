def solution(a, b):
    day_of_the_week = ["FRI", "SAT", "SUN", "MON", "TUE", "WED", "THU"]
    months = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days = 0

    for i in range(a) :
        days += months[i]
    days += b
    days = days % 7 - 1

    return day_of_the_week[days % 7]