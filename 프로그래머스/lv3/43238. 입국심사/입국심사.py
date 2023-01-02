def solution(n, times):
    start, end = 1, int(1e13)
    last = 0

    while start <= end:
        mid, result = (start + end) // 2, 0

        for time in times:
            result += mid // time

        if n > result:
            start = mid + 1
        else:
            last = mid
            end = mid - 1

    return last