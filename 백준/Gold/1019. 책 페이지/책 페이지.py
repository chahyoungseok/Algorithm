import copy


def cal_1(len_N) :
    data = [0] * 10
    for i in range(len_N, 0, -1):
        zero_a = int(9 * (len_N - i - 1) * (10 ** (len_N - i - 2)))
        data[0] += zero_a
        for j in range(1, 10):
            data[j] += int(zero_a + 10 ** (len_N - i - 1))
    return data


def cal_2(N) :
    list_N_2 = list(str(N))
    len_N_2 = len(list_N_2)
    stand = cal_1(len_N_2)
    total_2 = copy.deepcopy(stand)

    for i in range(1, int(list_N_2[0])):
        total_2[i] += 10 ** (len_N_2 - 1) - 1

        for j in list(str(i * (10 ** (len_N_2 - 1)))):
            total_2[int(j)] += 1

        for j in range(10):
            total_2[j] += stand[j]

        total_2[0] += ((len_N_2 - 1) * (10 ** (len_N_2 - 1) - 1) - sum(stand))

    for i in list(str(int(list_N_2[0]) * (10 ** (len_N_2 - 1)))):
        total_2[int(i)] += 1

    total_2[int(list_N_2[0])] += int("".join(list_N_2[1:]))

    return int("".join(list_N_2[1:])), total_2


N = int(input())
N_list = list(str(N))
N_len = len(N_list)

if N >= 10 :
    total_number = N_len * (N - (10 ** (N_len - 1) - 1))
    for i in range(1, N_len) :
        total_number += (N_len - i) * (10 ** (N_len - i - 1)) * 9
else :
    total_number = int(N_list[N_len - 1])

real_total = [0] * 10
while N >= 10 :
    N, total = cal_2(N)
    for i in range(10):
        real_total[i] += total[i]

for i in range(1, N + 1) :
    real_total[i] += 1

real_total[0] += total_number - sum(real_total)
for i in real_total :
    print(i, end=" ")