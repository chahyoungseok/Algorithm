string_1 = input()
string_2 = input()

len_1 = len(string_1)
len_2 = len(string_2)
lcs = [["" for _ in range(len_2 + 1)] for _ in range(len_1 + 1)]

for i in range(1, len_1 + 1):
    for j in range(1, len_2 + 1):
        if string_1[i - 1] == string_2[j - 1]:
            lcs[i][j] = lcs[i - 1][j - 1] + string_1[i - 1]
        else:
            lcs[i][j] = max(lcs[i - 1][j], lcs[i][j - 1], key=lambda x:len(x))

result = ""
for i in range(1, len_1 + 1):
    result = max(result, max(lcs[i], key=lambda x : len(x)), key=lambda x : len(x))

result_len = len(result)
print(result_len)
if result_len != 0:
    print(result)
