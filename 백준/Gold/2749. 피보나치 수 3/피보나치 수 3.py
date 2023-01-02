import sys

n = int(sys.stdin.readline().strip())
fi = [0, 1]
mod = 10**6
p = mod // 10 * 15
for i in range(2, p) :
    fi.append((fi[i - 1] + fi[i - 2]) % mod)

print(fi[n % p])