N = int(input())
if N % 4 == 0 :
    if N % 100 != 0 or N % 400 == 0 :
        print(1)
    else :
        print(0)
else :
    print(0)