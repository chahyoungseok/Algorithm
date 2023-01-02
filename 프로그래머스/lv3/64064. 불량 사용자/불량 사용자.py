import copy


def check_equal_id(id_1, id_2) :
    # id_1은 user_id, id_2는 banned_id
    id_1_len = len(id_1)
    if id_1_len != len(id_2) :
        return False

    for i in range(id_1_len) :
        if id_1[i] != id_2[i] and id_2[i] != '*' :
            return False
    return True


def solution(user_id, banned_id):
    answer, banned_id_len, banned_case, last_arr = 0, len(banned_id), [], []
    candidate_id = [[] for _ in range(banned_id_len)]

    for i in range(banned_id_len) :
        for user in user_id :
            if check_equal_id(user, banned_id[i]) :
                candidate_id[i].append(user)

    for i in candidate_id[0]:
        banned_case.append([i])

    for i in range(1, len(candidate_id)) :
        standard = copy.deepcopy(banned_case)
        for j in standard :
            for k in range(len(candidate_id[i])) :
                copy_j = copy.deepcopy(j)
                if candidate_id[i][k] in copy_j :
                    continue
                else :
                    copy_j.append(candidate_id[i][k])
                banned_case.append(copy_j)
            banned_case.pop(0)

    for i in range(len(banned_case)) :
        arr = set(banned_case[i])
        if arr not in last_arr and len(arr) == banned_id_len :
            last_arr.append(arr)
            answer += 1

    return answer