def change_alphabat(alphabat) :
    if 64 < ord(alphabat) < 91 :
        return chr(ord(alphabat) + 32)
    elif 96 < ord(alphabat) < 123 :
        return chr(ord(alphabat) - 32)

def solution(s):
    answer = ''
    arr = s.split(" ")
    for sel in range(len(arr)):
        list_sel = list(arr[sel])
        for i in range(len(list_sel)):
            if (i % 2 == 0 and 96 < ord(list_sel[i]) < 123) or (i % 2 == 1 and 64 < ord(list_sel[i]) < 91):
                list_sel[i] = change_alphabat(list_sel[i])
        arr[sel] = "".join(list_sel)

    for i in arr:
        answer += (i + " ")

    return "".join(answer[:-1])