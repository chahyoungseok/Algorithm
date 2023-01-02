from collections import deque


def solution(n, t, m, p):
    answer = ''
    result_list, trans_number = [], {10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F'}
    answer_index = deque()

    def get_N_Num(N, value, lists):
        a, b = value // N, value % N
        lists.append(b)
        if a == 0:
            return lists
        else:
            return get_N_Num(N, a, lists)

    answer_index.append(p - 1)
    for _ in range(t - 1):
        answer_index.append(m - 1)

    for i in range(10):
        trans_number[i] = str(i)

    number, q = 0, deque()
    while answer_index:
        if not q :
            q = deque(get_N_Num(n, number, []))
            q.reverse()
            while len(q) > 1 and q[0] == 0:
                q.popleft()

        while answer_index[0] + 1 > len(q) :
            answer_index[0] -= len(q)
            number += 1
            q = deque(get_N_Num(n, number, []))
            q.reverse()
            while len(q) > 1 and q[0] == 0:
                q.popleft()

        answer += trans_number[q[answer_index[0]]]

        for _ in range(answer_index[0] + 1) :
            q.popleft()

        if not q :
            number += 1
        answer_index.popleft()

    return answer