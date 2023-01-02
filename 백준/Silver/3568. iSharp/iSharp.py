import sys

data = sys.stdin.readline().strip().replace(";", "").replace(",", "").split(" ")

for i in range(1, len(data)) :
    print(data[0], end="")
    standard = 0
    for j in range(len(data[i]) - 1, 0, -1) :
        if not (data[i][j] == "[" or data[i][j] == "]" or data[i][j] == "*" or data[i][j] == "&") :
            standard = j
            break
        if data[i][j] == "]" :
            print("[", end="")
        elif data[i][j] == "[" :
            print("]", end="")
        else :
            print(data[i][j], end="")
    print(" ", end="")
    for k in range(standard + 1) :
        print(data[i][k], end="")
    print(";")