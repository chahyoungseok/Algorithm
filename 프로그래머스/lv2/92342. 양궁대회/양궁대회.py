import copy


def dfs(n, dist, case_dict, remain, lion_point, answer):
    global result, apeach_point, result_answer

    if lion_point > apeach_point:
        answer = sorted(answer)
        if lion_point - apeach_point > result:
            result_answer = []
            result = lion_point - apeach_point
            result_answer.append(answer)
        elif lion_point - apeach_point == result :
            if answer not in result_answer :
                result_answer.append(answer)

    if dist > n or remain == 0:
        return -1

    for case in case_dict.keys():
        if not case_dict[case] :
            continue
        copy_dict = copy.deepcopy(case_dict)
        target = copy_dict[case][0]
        copy_dict[case].remove(target)
        if remain - case >= 0:
            copy_answer = copy.deepcopy(answer)
            copy_answer.append(case)
            dfs(n, dist + 1, copy_dict, remain - case, lion_point + target, copy_answer)


result, result_answer, apeach_point = -1, [], 0


def solution(n, info):
    global apeach_point
    case_dict, answer_list = {}, []

    for i in range(11):
        if info[i] > 0 :
            apeach_point += (10 - i)

        if info[i] + 1 != 1:
            point = (10 - i) * 2
        else:
            point = (10 - i)

        if (info[i] + 1) not in case_dict.keys():
            case_dict[info[i] + 1] = [point]
        else :
            case_dict[info[i] + 1].append(point)

    for case in case_dict.keys():
        copy_dict, remain = copy.deepcopy(case_dict), n
        target = copy_dict[case][0]
        copy_dict[case].remove(target)
        if remain - case >= 0:
            dfs(n, 1, copy_dict, remain - case, target, [case])

    if result == -1 :
        return [-1]
    else :
        for arr in result_answer :
            copy_info = copy.deepcopy(info)
            answer = [0 for _ in range(11)]
            for i in arr :
                for j in range(11) :
                    if copy_info[j] == i - 1 :
                        answer[j] = info[j] + 1
                        copy_info[j] = -1
                        break

            answer_list.append(answer)

    if len(answer_list) >= 2 :
        for i in range(10, -1, -1) :
            max_depth = 0
            for j in range(len(answer_list)) :
                max_depth = max(max_depth, answer_list[j][i])

            for j in range(len(answer_list) - 1, -1, -1) :
                if max_depth > answer_list[j][i] :
                    answer_list.pop(j)

            if len(answer_list) == 1 :
                break
    
    answer = answer_list[0]
    answer[10] += n - sum(answer)

    return answer