from itertools import permutations

def calculation(number_list, oper) :
    sum = number_list[0]
    for i in range(len(oper)) :
        if oper[i] == '+':
            sum += number_list[i + 1]
        elif oper[i] == '-':
            sum -= number_list[i + 1]
        elif oper[i] == '*':
            sum *= number_list[i + 1]
        else:
            if sum > 0:
                sum //= number_list[i + 1]
            else:
                sum = ((sum * -1) // number_list[i + 1]) * -1

    return sum


N = int(input())
number_list = list(map(int,input().split()))
oper_list = list(map(int, input().split()))
opers = ['+','-','*','/']
liner_oper_list = []
for i in range(len(oper_list)) :
    for j in range(oper_list[i]) :
        liner_oper_list.append(opers[i])

max_result = 0
min_result = int(1e9)

for oper in permutations(liner_oper_list,len(liner_oper_list)) :
    result = calculation(number_list, oper)
    if result > max_result :
        max_result = result
    if result < min_result :
        min_result = result

print(max_result)
print(min_result)