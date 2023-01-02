def solution(s):
    max_sel, min_sel, arr = -int(1e9), int(1e9), s.split(' ')
    for sel in arr :
        sel_int = int(sel)
        if sel_int > max_sel :
            max_sel = sel_int
        if min_sel > sel_int :
            min_sel = sel_int
    answer = str(min_sel) + " " + str(max_sel)
    return answer