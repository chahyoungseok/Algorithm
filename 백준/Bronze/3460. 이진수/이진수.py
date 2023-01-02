T = int(input())
for _ in range(T) :
    n = int(input())
    l = []
    while n != 0 :
        l.append(n % 2)
        n //= 2

    for i in range(len(l)) :
        if l[i] == 1 :
            print(i, end=" ")