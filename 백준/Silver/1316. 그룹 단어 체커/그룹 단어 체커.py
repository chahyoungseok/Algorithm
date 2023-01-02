import sys

N = int(input())

total = 0
for _ in range(N) :
    overlap_confirm, state = [], True
    data = sys.stdin.readline().strip()
    standard = data[0]
    for i in range(1, len(data)) :
        if standard == data[i] :
            continue
        else :
            if standard in overlap_confirm :
                state = False
                break
            else :
                overlap_confirm.append(standard)
            standard = data[i]

    if standard in overlap_confirm :
        state = False

    if state :
        total += 1
print(total)