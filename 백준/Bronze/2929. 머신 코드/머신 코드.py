import sys, re

data = sys.stdin.readline().strip()
answer = 0
results = re.split('[A-Z]', data)
for i in range(1, len(results) - 1) :
    answer += (4 - ((len(results[i]) + 1) % 4)) % 4
print(answer)