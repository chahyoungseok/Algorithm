import re
expression = input()
ex_sum = 0

expression = re.split("([-|+])", expression)
expression_len = len(expression)
index = expression_len

for i in range(len(expression)) :
    if expression[i] == "-":
        index = i
        break

for i in range(index + 1, expression_len, 2) :
    ex_sum -= int(expression[i])

for i in range(0, index, 2) :
    ex_sum += int(expression[i])

print(ex_sum)