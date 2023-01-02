import sys, re

data = ""
while True :
    data = sys.stdin.readline().strip()
    if data == "EOI" :
        break

    data = data.lower()
    print("Found" if re.search('nemo', data) else "Missing")