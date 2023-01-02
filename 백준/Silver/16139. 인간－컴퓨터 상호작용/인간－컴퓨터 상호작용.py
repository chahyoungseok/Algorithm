import sys
input = sys.stdin.readline

S = input().strip()
q = int(input())
a_asc = ord('a')
dp_list = [[0 for _ in range(len(S) + 1)] for _ in range(26)]

for i in range(len(S)):
    dp_list[ord(S[i]) - a_asc][i + 1] = 1

for i in range(len(S)):
    for j in range(26) :
        dp_list[j][i + 1] += dp_list[j][i]

for _ in range(q) :
    a, l, r = map(str, input().split())
    print(dp_list[ord(a) - a_asc][int(r) + 1] - dp_list[ord(a) - a_asc][int(l)])
