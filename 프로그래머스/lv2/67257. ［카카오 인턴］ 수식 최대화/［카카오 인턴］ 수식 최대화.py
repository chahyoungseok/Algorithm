from itertools import permutations
import re, copy


def solution(expression):
    expression, oper = re.split('([-|+|*])', expression), ['+', '-', '*']
    answer = 0

    for combin in permutations(oper,3) :
        expression_copy = copy.deepcopy(expression)
        for oper_case in combin :
            cal_arr = []
            i = 0
            while len(expression_copy) > i :
                if expression_copy[i] == oper_case :
                    cal_arr.append(str(eval(cal_arr.pop(len(cal_arr) - 1) + expression_copy[i] + expression_copy[i + 1])))
                    i += 1
                else :
                    cal_arr.append(expression_copy[i])
                i += 1

            expression_copy = cal_arr
        answer = max(answer, abs(int(expression_copy[0])))

    return answer