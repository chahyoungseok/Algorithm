import sys

N = int(input())
file_list, result = [], []
for _ in range(N) :
    file_list.append(sys.stdin.readline().strip())

for i in range(len(file_list[0])) :
    standard, state = file_list[0][i], True
    for j in range(1, N) :
        if standard != file_list[j][i] :
            state = False
            break
    if state :
        result.append(standard)
    else :
        result.append("?")
print("".join(result))