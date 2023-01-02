import copy, sys


def cal(current, dist, c_op) :
    global max_value, min_value
    if dist == N:
        max_value = max(max_value, current)
        min_value = min(min_value, current)
        return

    for o in range(4) :
        if o in c_op :
            copy_op = copy.deepcopy(c_op)
            copy_op.remove(o)
            if o != 3 or current >= 0:
                cal(eval(str(current) + oper[o] + str(A_list[dist])), dist + 1, copy_op)
            else:
                cal(((current * -1) // A_list[dist]) * -1, dist + 1, copy_op)


N = int(input())
A_list = list(map(int, (sys.stdin.readline()).split()))
op_list = list(map(int, (sys.stdin.readline()).split()))
op = []
for i in range(4) :
    for j in range(op_list[i]) :
        op.append(i)
oper = ['+', '-', '*', '//']
max_value, min_value = -int(1e10), int(1e10)

cal(A_list[0], 1, op)
print(max_value)
print(min_value)