N = int(input())
pre_one, pre_zero = 1, 0
one, zero = 1, 0
for i in range(N - 1) :
    zero = pre_one + pre_zero
    one = pre_zero

    pre_one, pre_zero = one, zero

print(zero + one)