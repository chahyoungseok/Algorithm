def LCM(sel1, sel2) :
    sel1_m = sel1
    sel2_m = sel2
    while sel1_m != sel2_m :
        if sel1_m > sel2_m :
            sel2_m += sel2
        else :
            sel1_m += sel1
    
    return sel1_m

def solution(arr):
    arr_len = len(arr) - 1
    
    for i in range(arr_len) :
        arr[i + 1] = LCM(arr[i + 1], arr[i])
    return arr[arr_len]