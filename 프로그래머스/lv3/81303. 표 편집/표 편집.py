def solution(n, k, cmd):
    linked_list, remove_stack = {i: [i - 1, i + 1] for i in range(n)}, []

    cur = k
    for case in cmd:
        data = case.split()
        if data[0] == "D":
            for _ in range(int(data[1])):
                cur = linked_list[cur][1]
        elif data[0] == "U":
            for _ in range(int(data[1])):
                cur = linked_list[cur][0]
        elif data[0] == "C":
            pre, nex = linked_list[cur]
            remove_stack.append([pre, cur, nex])
            if pre == -1:
                linked_list[nex][0] = pre
            elif nex == n:
                linked_list[pre][1] = nex
            else :
                linked_list[nex][0] = pre
                linked_list[pre][1] = nex

            if nex not in linked_list.keys():
                cur = pre
            else :
                cur = nex
        elif data[0] == "Z":
            pre, now, nex = remove_stack.pop()
            if pre == -1:
                linked_list[nex][0] = now
            elif nex == n:
                linked_list[pre][1] = now
            else :
                linked_list[nex][0] = now
                linked_list[pre][1] = now
    answer = ["O" for _ in range(n)]
    for pre, now, nex in remove_stack :
        answer[now] = "X"

    return "".join(answer)