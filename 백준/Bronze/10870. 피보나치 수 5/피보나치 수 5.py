n = int(input())
fa = [0, 1]
for i in range(2, n + 1) :
    fa.append(fa[i - 1] + fa[i - 2])
print(fa[n])