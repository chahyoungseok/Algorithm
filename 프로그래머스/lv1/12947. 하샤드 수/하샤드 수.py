def solution(x):
    x, number = list(str(x)), 0
    for i in x:
        number += int(i)

    if int("".join(x)) % number == 0:
        return True
    return False