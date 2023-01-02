import sys

N = int(sys.stdin.readline().strip())

tmp, count = str(N), 1
if N < 10 :
    tmp += "0"

next_tmp = str(int(tmp[0]) + int(tmp[1]))
if N < 10 :
    tmp = str(N) + next_tmp[-1]
else :
    tmp = tmp[1] + next_tmp[-1]

while int(tmp) != N :
    next_tmp = str(int(tmp[0]) + int(tmp[1]))
    tmp = tmp[1] + next_tmp[-1]

    count += 1
print(count)
