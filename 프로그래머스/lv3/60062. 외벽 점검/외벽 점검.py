from itertools import permutations
from collections import deque
import copy


def solution(n, weak, dist):
    answer = -1
    for people_len in range(1, len(dist) + 1):

        def check(n, weak, people):
            weak, weak_len = deque(weak), len(weak)
            for c in range(weak_len):
                # print("weak : ", weak)
                for perm in permutations(people):
                    # print("perm", perm)
                    copy_weak = copy.deepcopy(weak)
                    for r in range(c):
                        copy_weak[weak_len - 1 - r] += n
                    # print("copy_weak", copy_weak)
                    perm_queue, weak_queue = deque(perm), deque(copy_weak)

                    while perm_queue:
                        point = weak_queue.popleft()
                        point += perm_queue.popleft()

                        while weak_queue and point >= weak_queue[0]:
                            weak_queue.popleft()

                            if not weak_queue:
                                return True
                        if not weak_queue:
                            return True

                weak.append(weak.popleft())

        if check(n, weak, dist[-people_len:]):
            answer = people_len
            break

    return answer