import sys, re

r = re.compile('(pi|ka|chu)+')
answer = r.fullmatch(sys.stdin.readline().strip())

print("YES" if answer != None else "NO")