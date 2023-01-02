N, K = map(int, input().split())

index, save_i = 0, 0

for i in range(1, N + 1) :
    if N % i == 0 :
        index += 1
        if index == K :
            save_i = i
            break
if save_i != 0 :
    print(save_i)
else :
    print(0)