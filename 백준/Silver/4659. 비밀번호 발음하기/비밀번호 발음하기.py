import sys, re

while True :
    data = sys.stdin.readline().strip()
    if data == "end" :
        break
    if not re.search('a|e|i|o|u', data) :
        print("<" + data + "> is not acceptable.")
        continue
    if re.search('([a|e|i|o|u]{3})|([^a|e|i|o|u]{3})', data) :
        print("<" + data + "> is not acceptable.")
        continue
    if re.search(r'([a-df-np-z])\1', data) :
        print("<" + data + "> is not acceptable.")
        continue
    print("<" + data + "> is acceptable.")