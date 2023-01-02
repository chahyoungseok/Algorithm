def solution(files):
    answer, file_spearate = [], []

    standard_0, standard_9, index = ord('0'), ord('9'), 0
    for file in files:
        file = file.lower()
        spearate = []

        head = 0
        for i in range(len(file)):
            head = i
            if standard_0 <= ord(file[i]) <= standard_9:
                break
        spearate.append(file[:head])

        body = len(file)
        for i in range(head, len(file)):
            target = ord(file[i])
            if standard_0 > target or target > standard_9:
                body = i
                break
        spearate.append(file[head: body])
        spearate.append(file[body:])

        file_spearate.append([index, spearate])
        index += 1

    result = sorted(file_spearate, key=lambda x: (x[1][0], int(x[1][1])))
    for number, spearated in result :
        answer.append(files[number])
    return answer