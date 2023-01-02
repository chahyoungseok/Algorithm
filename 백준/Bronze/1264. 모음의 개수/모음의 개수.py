import sys, re

data = ""
while True :
    data = sys.stdin.readline().strip().lower()
    if data == "#" :
        break

    result = re.findall('a|e|i|o|u', data)
    print(len(result))