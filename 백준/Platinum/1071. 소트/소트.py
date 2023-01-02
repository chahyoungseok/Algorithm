import sys
from collections import deque

N = int(sys.stdin.readline().strip())
numbers = list(map(int, (sys.stdin.readline()).split()))
numbers, number_dict = sorted(numbers), {}
q = deque(numbers)

for i in range(N) :
    target = numbers[i]
    if target in number_dict.keys() :
        number_dict[target] += 1
    else :
        number_dict[target] = 1

while q :
    target = q.popleft()
    if target + 1 in number_dict.keys() :
        state = True
        for j in number_dict.keys() :
            if j > target + 1 :
                state = False
                for _ in range(number_dict[target]) :
                    print(target, end=" ")
                for _ in range(number_dict[target] - 1) :
                    q.popleft()
                print(j, end=" ")
                q.remove(j)
                del number_dict[target]
                if number_dict[j] == 1 :
                    del number_dict[j]
                else :
                    number_dict[j] -= 1
                break
        if state :
            for _ in range(number_dict[target + 1]):
                q.popleft()
            for _ in range(number_dict[target + 1]) :
                print(target + 1, end=" ")
            for _ in range(number_dict[target] - 1):
                q.popleft()
            for _ in range(number_dict[target]):
                print(target, end=" ")
            del number_dict[target], number_dict[target + 1]
    else :
        for _ in range(number_dict[target]) :
            print(target, end=" ")
        for _ in range(number_dict[target] - 1):
            q.popleft()
        del number_dict[target]
