import sys

N = int(sys.stdin.readline().strip())
F = int(sys.stdin.readline().strip())

state = ""
N -= N % 100
for i in range(100) :
    if (N + i) % F == 0 :
        state = str(i)
        break

if len(state) == 1:
    print("0" + state)
else :
    print(state)