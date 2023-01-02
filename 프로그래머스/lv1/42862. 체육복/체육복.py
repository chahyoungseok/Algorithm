def solution(n, lost, reserve):
    set_reserve = set(reserve) - set(lost)
    set_lost = set(lost) - set(reserve)
    for number in set_lost:
        if number - 1 in set_reserve:
            set_reserve.remove(number - 1)
        elif number + 1 in set_reserve:
            set_reserve.remove(number + 1)
        else :
            n-=1

    return n