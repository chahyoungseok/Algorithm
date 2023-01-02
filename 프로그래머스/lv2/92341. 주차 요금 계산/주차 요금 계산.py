from collections import defaultdict


def solution(fees, records):
    record_split, records_len = [], len(records)
    fee_dict, pre_state = defaultdict(int), "OUT"
    answer = []

    for i in range(records_len) :
        record_split.append(records[i].split(" "))
        time = record_split[i][0].split(":")
        record_split[i][0] = int(time[0]) * 60 + int(time[1])

    record_split = sorted(record_split, key=lambda x:int(x[1]))

    for record in record_split :
        time, car_number, state = record

        if state == "IN":
            if pre_state == "IN":
                fee_dict[pre_car_number] += (1439 - pre_time)
        else :
            fee_dict[pre_car_number] += (time - pre_time)

        pre_time, pre_car_number, pre_state = time, car_number, state

    if state == "IN":
        fee_dict[pre_car_number] += (1439 - pre_time)

    for fee in fee_dict.items() :
        extra_time = fee[1] - fees[0]
        if extra_time <= 0 :
            answer.append(fees[1])
        else :
            time_unit = extra_time // fees[2]
            if extra_time % fees[2] != 0 :
                time_unit += 1
            answer.append(time_unit * fees[3] + fees[1])

    return answer