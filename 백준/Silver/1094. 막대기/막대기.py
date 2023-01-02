import sys

X = int(sys.stdin.readline().strip())
if X == 64 :
    print(1)
else :
    sticks = [64]
    
    while True :
        min_half_stick = sticks.pop() // 2
        sticks.append(min_half_stick)
        stick_sum = sum(sticks)
        if X > stick_sum:
            sticks.append(min_half_stick)
        elif X == stick_sum:
            print(len(sticks))
            break