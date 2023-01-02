import sys

N = int(input())
ls = list(map(int, (sys.stdin.readline()).split()))
print(str(min(ls)) + " " + str(max(ls)))