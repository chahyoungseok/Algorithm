from collections import deque

N, K = map(int, input().split())

examiner, virus_storage = [], []

for i in range(N) :
    examiner_part = list(map(int, input().split()))
    examiner.append(examiner_part)
    for j in range(N) :
        if examiner_part[j] != 0 :
            virus_storage.append((examiner_part[j], i, j, 0))

S, X, Y = map(int, input().split())
virus_storage = sorted(virus_storage)
queue = deque(virus_storage)

while queue:
    virus_number, row, colum, dist = queue.popleft()
    if dist >= S:
        break

    if row - 1 >= 0 and examiner[row - 1][colum] == 0:
        examiner[row - 1][colum] = virus_number
        queue.append((virus_number, row - 1, colum, dist + 1))
    if row + 1 < N and examiner[row + 1][colum] == 0:
        examiner[row + 1][colum] = virus_number
        queue.append((virus_number, row + 1, colum, dist + 1))
    if colum - 1 >= 0 and examiner[row][colum - 1] == 0:
        examiner[row][colum - 1] = virus_number
        queue.append((virus_number, row, colum - 1, dist + 1))
    if colum + 1 < N and examiner[row][colum + 1] == 0:
        examiner[row][colum + 1] = virus_number
        queue.append((virus_number, row, colum + 1, dist + 1))

print(examiner[X - 1][Y - 1])