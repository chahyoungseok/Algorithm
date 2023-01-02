import sys

T = int(sys.stdin.readline().strip())
for _ in range(T) :
    inp = sys.stdin.readline().strip()
    print(str(inp[0]) + str(inp[len(inp) - 1]))