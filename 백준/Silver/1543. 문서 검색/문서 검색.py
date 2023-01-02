import re

A = input()
B = input()

count = 0
while True :
    result = re.search(B, A)
    if not result :
        break
    a, b = result.span()
    A = A[b:]
    count += 1

print(count)