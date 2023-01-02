A, B, V = map(int, input().split())
if V == A :
    print(1)
else :
    index = int((V - B) / (A - B))
    if (V - B) % (A - B) == 0 :
        print(index)
    else :
        print(index + 1)