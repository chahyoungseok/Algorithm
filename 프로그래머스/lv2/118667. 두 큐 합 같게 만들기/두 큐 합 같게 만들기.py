from collections import deque


def solution(queue1, queue2):
    sum_q_1, sum_q_2 = sum(queue1), sum(queue2)
    answer, queue_length = 0, len(queue1)
    queue1, queue2 = deque(queue1), deque(queue2)
    
    while sum_q_1 != sum_q_2:
        if sum_q_1 > sum_q_2:
            target_element = queue1.popleft()
            queue2.append(target_element)

            sum_q_2 += target_element
            sum_q_1 -= target_element
        else:
            target_element = queue2.popleft()
            queue1.append(target_element)

            sum_q_1 += target_element
            sum_q_2 -= target_element
        answer += 1

        if answer > 3 * queue_length:
            answer = -1
            break

    return answer