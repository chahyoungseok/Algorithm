from itertools import combinations

def in_check(arr1, arr2) :
    arr1_len = len(arr1)
    result = 0
    for sel in arr1 :
        if sel in arr2 :
            result += 1

    if arr1_len == result :
        return False
    else :
        return True


def solution(relation):
    answer = []
    row, col = len(relation), len(relation[0])
    combination_index, overlap_set = [], []

    for i in range(1, col + 1) :
        combination_index.extend(combinations(range(col), i))

    for combin in combination_index :
        for i in range(row) :
            tup = ()
            for c in combin :
                tup += (relation[i][c],)
            overlap_set.append(tup)

        if len(set(overlap_set)) == row :
            state = True
            for complete in answer :
                state = in_check(complete, combin)
                if not state :
                    break
            if state:
                answer += (combin,)
        overlap_set = []

    return len(answer)