import copy

N = int(input())
total = [1] * 10
total[0] = 0
for _ in range(1, N) :
    pre_total = copy.deepcopy(total)
    total = [0] * 10
    for i in range(10) :
        if i == 0 :
            total[1] += pre_total[0]
        elif i == 9 :
            total[8] += pre_total[9]
        else :
            total[i + 1] += pre_total[i]
            total[i - 1] += pre_total[i]

print(sum(total) % 1000000000)