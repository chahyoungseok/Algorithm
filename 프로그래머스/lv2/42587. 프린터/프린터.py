from collections import deque

def check_priorities(priorities):
    first = priorities[0]
    for i in range(1, len(priorities)):
        if priorities[i] > first:
            return False
    return True


def solution(priorities, location):
    priorities = deque(priorities)
    pop_index = 0

    while True:
        while not check_priorities(priorities):
            priorities.rotate(-1)
            location -= 1
            if location < 0:
                location = len(priorities) - 1

        if location <= 0 and check_priorities(priorities) :
            break
        priorities.popleft()
        location -= 1
        pop_index += 1
        if location < 0:
            location = len(priorities) - 1

    return pop_index + 1