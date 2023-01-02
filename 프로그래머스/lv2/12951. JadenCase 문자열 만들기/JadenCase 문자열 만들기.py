def solution(s):
    arr = list(s)
    if 96 < ord(arr[0]) < 123:
        arr[0] = chr(ord(arr[0]) - 32)

    for i in range(1, len(arr)):
        if arr[i - 1] == " " and 96 < ord(arr[i]) < 123:
            arr[i] = chr(ord(arr[i]) - 32)
        elif arr[i - 1] != " " and 64 < ord(arr[i]) < 91 :
            arr[i] = chr(ord(arr[i]) + 32)

    return "".join(arr)