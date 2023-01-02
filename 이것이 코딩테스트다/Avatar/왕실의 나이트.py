def numberOfCase(canMove, RC) :
    if RC == 1 or RC == 8:
        canMove /= 2
    elif RC == 2 or RC == 7:
        if canMove % 4 == 0:
            canMove *= 3 / 4
        else:
            canMove *= 2 / 3

    return canMove

loc = input()

row = int(loc[1])
col = ord(loc[0]) - ord('a') + 1

canMove = 8
canMove = numberOfCase(canMove,row)
canMove = numberOfCase(canMove,col)

print(int(canMove))