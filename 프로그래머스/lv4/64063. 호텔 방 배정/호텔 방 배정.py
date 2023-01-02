from collections import defaultdict


def solution(k, room_number):
    answer = []
    room = defaultdict(int)

    for number in room_number:
        if not room[number] :
            room[number] = number + 1
            answer.append(number)
        else:
            list = [number]
            index = number
            while room[index] :
                index = room[index]
                list.append(index)

            for i in list:
                room[i] = index + 1

            room[index] = index + 1
            room[number] = index + 1
            answer.append(index)

    return answer