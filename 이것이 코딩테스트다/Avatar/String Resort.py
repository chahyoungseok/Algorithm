S = input()
n_sum = 0
q = []

for i in S :
    if i.isalpha() :
        q.append(i)
    else :
        n_sum += int(i)

q.sort()

if n_sum != 0 :
    q.append(str(n_sum))

print(''.join(q))