from collections import deque

word = input()
q, total = deque(), 0
c_2 = ["c=", "c-", "d-", "lj", "nj", "s=", "z="]
c_3 = "dz="
for i in word :
    q.append(i)
    if len(q) == 2 :
        if "".join(list(q)) in c_2 :
            q.clear()
            total += 1
    if len(q) == 3 :
        if "".join(list(q)) == c_3 :
            q.clear()
            total += 1
        else :
            q.popleft()
            total += 1
            if "".join(list(q)) in c_2:
                q.clear()
                total += 1

if q :
    total += len(q)

print(total)
