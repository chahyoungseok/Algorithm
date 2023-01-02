def solution(s, n):
    list_s = list(s)

    for i in range(len(list_s)):
        ord_s = ord(list_s[i])
        if (64 < ord_s < 91) or (96 < ord_s < 123):
            trans_s = ord_s + n
        else :
            list_s[i] = ' '
            continue

        if ord_s > 96:
            if trans_s > 122:
                trans_s -= 26
        elif ord_s > 64:
            if trans_s > 90:
                trans_s -= 26

        list_s[i] = chr(trans_s)

    return "".join(list_s)