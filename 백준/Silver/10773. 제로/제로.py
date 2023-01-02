K = int(input())
stack = []

for _ in range(K) :
    data = int(input())
    if data == 0 :
        stack.pop(len(stack) - 1)
    else :
        stack.append(data)

print(sum(stack))