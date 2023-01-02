def is_VPS(data) :
    while "()" in data :
        data = data.replace("()", "")

    if data == "" :
        return "YES"
    else:
        return "NO"


T = int(input())

for _ in range(T) :
    print(is_VPS(input()))