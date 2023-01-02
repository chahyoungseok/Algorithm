import sys

alp_dict = {chr(c) : 0 for c in range(ord('A'), ord('Z') + 1)}

str = sys.stdin.readline().strip().upper()
for char in str :
    alp_dict[char] += 1

equal_state = False
max_value, max_count = '?', 0
for key in alp_dict.keys() :
    if alp_dict[key] > max_count :
        max_count = alp_dict[key]
        max_value = key
        equal_state = False
    elif alp_dict[key] == max_count :
        equal_state = True

if equal_state :
    print("?")
else :
    print(max_value)