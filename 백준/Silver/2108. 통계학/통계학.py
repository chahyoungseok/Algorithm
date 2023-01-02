import copy
import sys
from collections import Counter

N = int(input())
data = []
for _ in range(N) :
    data.append(int((sys.stdin.readline()).strip()))

data = sorted(data)
counter, max_number, sub_number = Counter(data), [], []
for i in counter.keys() :
    if not max_number :
        max_number = [i, counter[i]]
    else :
        if max_number[1] >= counter[i] :
            if not sub_number :
                sub_number = [i, counter[i]]
            else :
                if counter[i] > sub_number[1] :
                    sub_number = [i, counter[i]]
        else :
            sub_number = copy.deepcopy(max_number)
            max_number = [i, counter[i]]

print(round(sum(data) / N))
print(data[N // 2])
if sub_number and max_number[1] == sub_number[1] :
    print(sub_number[0])
else :
    print(max_number[0])
print(data[N - 1] - data[0])