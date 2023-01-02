def makeNumber(number) :
    result = number
    for i in str(number) :
        result += int(i)

    return result


non_selfnumber = set()
for i in range(1, 10000) :
    pre_i = makeNumber(i)
    if 10000 >= pre_i :
        non_selfnumber.add(pre_i)

non_selfnumber = sorted(non_selfnumber)
for i in range(1, 10000) :
    if i not in non_selfnumber :
        print(i)