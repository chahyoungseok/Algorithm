import sys

data = sys.stdin.readline().strip()
data = list(data.replace('()', '2').replace('[]', '3'))
s_arr, state, pre_N = ['(',')','[',']'], True, -1

while state :
    N, state = len(data), False
    if N == pre_N or (N == 1 and data[0] in s_arr):
        print(0)
        break
    elif N == 1 :
        print(int(data[0]))
        break

    for i in range(N):
        if data[i] not in s_arr:
            state = True
            if i - 1 >= 0 and i + 1 < N :
                if data[i - 1] == '(' and data[i + 1] == ')' :
                    data[i - 1] = str(int(data[i]) * 2)
                    data.pop(i + 1)
                    data.pop(i)
                    break
                elif data[i - 1] == '[' and data[i + 1] == ']' :
                    data[i - 1] = str(int(data[i]) * 3)
                    data.pop(i + 1)
                    data.pop(i)
                    break
                elif data[i + 1] not in s_arr :
                    data[i] = str(int(data[i]) + int(data[i + 1]))
                    data.pop(i + 1)
                    break
                else :
                    continue
            else :
                if i + 1 < N and data[i + 1] not in s_arr:
                    data[i] = str(int(data[i]) + int(data[i + 1]))
                    data.pop(i + 1)
                    break
    pre_N = N
    if not state :
        print(0)
        break