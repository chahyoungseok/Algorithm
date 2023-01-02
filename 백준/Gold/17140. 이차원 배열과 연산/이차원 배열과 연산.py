import copy, sys

r, c, k = map(int, (sys.stdin.readline()).split())
A = []
c_R, c_C, time = 3, 3, 0
for _ in range(3) :
    A.append(list(map(int, (sys.stdin.readline()).split())))

while time <= 100 :
    if len(A) > r - 1 and len(A[0]) > c - 1 and A[r - 1][c - 1] == k :
        print(time)
        break

    tmp = []
    if c_R >= c_C :
        # R 연산
        max_length = 0
        for i in range(c_R) :
            count_dict, result = {}, []
            for j in range(c_C) :
                if A[i][j] == 0 :
                    continue
                if A[i][j] in count_dict.keys() :
                    count_dict[A[i][j]] += 1
                else :
                    count_dict[A[i][j]] = 1

            key_list = sorted(list(count_dict.keys()), key=lambda x : (count_dict[x], x))
            for key in key_list :
                result.append(key)
                result.append(count_dict[key])

            if len(result) > max_length :
                max_length = len(result)

            tmp.append(result)

        for i in range(c_R) :
            tmp_i_len = len(tmp[i])
            if max_length > tmp_i_len :
                for j in range(max_length - tmp_i_len) :
                    tmp[i].append(0)
        A = copy.deepcopy(tmp)
        c_C = max_length
    else :
        # C 연산
        max_length = 0
        for i in range(c_C):
            count_dict, result = {}, []
            for j in range(c_R):
                if A[j][i] == 0:
                    continue
                if A[j][i] in count_dict.keys():
                    count_dict[A[j][i]] += 1
                else:
                    count_dict[A[j][i]] = 1

            key_list = sorted(list(count_dict.keys()), key=lambda x: (count_dict[x], x))
            for key in key_list:
                result.append(key)
                result.append(count_dict[key])

            if len(result) > max_length:
                max_length = len(result)

            tmp.append(result)

        for i in range(c_C):
            tmp_i_len = len(tmp[i])
            if max_length > tmp_i_len:
                for j in range(max_length - tmp_i_len):
                    tmp[i].append(0)

        trans_tmp = []
        for i in range(max_length) :
            rows = []
            for j in range(len(tmp)) :
                rows.append(tmp[j][i])
            trans_tmp.append(rows)
        A = copy.deepcopy(trans_tmp)
        c_R = max_length
    time += 1

if time > 100 :
    print(-1)