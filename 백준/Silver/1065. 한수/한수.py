def check(number) :
    n_l = list(str(number))
    n_len = len(n_l)
    if n_len > 2 :
        s = int(n_l[1]) - int(n_l[0])
        for i in range(2, n_len) :
            if int(n_l[i]) - int(n_l[i - 1]) != s :
                return False

    return True


X = int(input())
total = 0
for i in range(1, X + 1) :
    if check(i) :
        total += 1

print(total)