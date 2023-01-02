import sys

n = int(sys.stdin.readline().strip())
fi = [0] * 91
fi[0], fi[1] = 0, 1
for i in range(2, 91) :
    fi[i] = fi[i - 1] + fi[i - 2]

print(fi[n])