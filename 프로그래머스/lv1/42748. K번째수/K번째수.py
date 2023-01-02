def solution(array, commands):
    answer = []
    for command in commands:
        cur_arr = array[command[0] - 1: command[1]]
        cur_arr.sort()
        answer.append(cur_arr[command[2] - 1])

    return answer