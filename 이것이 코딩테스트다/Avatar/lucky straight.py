N = input()
half_index = int(len(N) / 2)

left_sum = 0
right_sum = 0

for i in range(0, half_index) :
    left_sum += int(N[i])
    right_sum += int(N[i + half_index])

if left_sum == right_sum :
    print("LUCKY")
else :
    print("READY")