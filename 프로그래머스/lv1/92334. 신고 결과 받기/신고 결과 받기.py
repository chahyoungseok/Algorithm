def solution(id_list, report, k):
    id_dict = {id: [] for id in id_list}
    result = {id: 0 for id in id_list}

    for case in report:
        reporting, reported = case.split()
        if reporting not in id_dict[reported]:
            id_dict[reported].append(reporting)

    for value in id_dict.values() :
        if len(value) >= k :
            for i in value :
                result[i] += 1

    return list(result.values())