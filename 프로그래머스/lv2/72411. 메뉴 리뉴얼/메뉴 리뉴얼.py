from itertools import combinations
from collections import Counter

def solution(orders, course):
    answer = []
    for course_case in course :
        menu_can, max_orders = [], 0
        for order in orders :
            for combin in combinations(order, course_case) :
                menu_can.append("".join(sorted(combin)))

        can_list = Counter(menu_can).most_common()
        if not can_list :
            continue
        max_orders = can_list[0][1]
        if max_orders < 2 :
            continue
        for can in  can_list:
            if can[1] != max_orders :
                break
            answer.append(can[0])

    return sorted(answer)