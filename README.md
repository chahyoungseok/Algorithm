# Algorithm

#### LAN Cable Cut

``` lan cable cut

lan_list = []
K, N = map(int, input().split())
for _ in range(K) :
    lan_list.append(int(input()))


def binary_search(target, start, end) :
    mid, sum = (start + end) // 2, 0
    if start > end :
        return mid

    for lan in lan_list :
        sum += lan // mid

    if target > sum :
        return binary_search(target, start, mid - 1)
    elif target <= sum :
        return binary_search(target, mid + 1, end)

print(binary_search(N, 1, max(lan_list)))
```

<br>

#### Black Jak

``` black jak
from itertools import combinations

N, M = map(int, input().split())
cards = list(map(int, input().split()))
close_M = int(1e9)

for combin in combinations(cards, 3) :
    combin_sum = sum(combin)
    if M >= combin_sum and close_M > M - combin_sum:
        close_M = M - combin_sum

print(M - close_M)
```

<br>

#### Gas Station

``` gas station
city_len = int(input())
city_between = list(map(int, input().split()))
city_oil = list(map(int, input().split()))

oil_sum = 0

record_oil = [[city_oil[city_len - 2], city_between[city_len - 2]]] #비용, 거리
for i in range(city_len - 3, -1, -1) :
    while record_oil and record_oil[len(record_oil) - 1][0] >= city_oil[i] :
        city_between[i] += record_oil[len(record_oil) - 1][1]
        record_oil.pop(len(record_oil) - 1)

    if not record_oil :
        record_oil = [[city_oil[i], city_between[i]]]
    else :
        record_oil.append([city_oil[i], city_between[i]])


for money, distance in record_oil :
    oil_sum += money * distance

print(oil_sum)
```

<br>

#### Lost Parenthesis

``` lost parenthesis
import re
expression = input()
ex_sum = 0

expression = re.split("([-|+])", expression)
expression_len = len(expression)
index = expression_len

for i in range(len(expression)) :
    if expression[i] == "-":
        index = i
        break

for i in range(index + 1, expression_len, 2) :
    ex_sum -= int(expression[i])

for i in range(0, index, 2) :
    ex_sum += int(expression[i])

print(ex_sum)
```

<br>

#### ATM

``` ATM
N = int(input())
P = list(map(int, input().split()))

time, total_time = 0, 0
P = sorted(P)

for i in P :
    time += i
    total_time += time

print(total_time)
```

<br>

#### Coin

``` coin
N, M = map(int, input().split())

money_case, total = [], 0
for _ in range(N) :
    money_case.append(int(input()))

money_case.reverse()

for money in money_case :
    sel = M // money
    total += sel
    M -= sel * money

print(total)
```

<br>

#### Conference Room Assignment

``` conference room assignment
N = int(input())

schedule, min_schedules, real_schedule, count = [], [], [], 0
for _ in range(N) :
    start, end = map(int, input().split())
    if start == end :
        count += 1
    else :
        schedule.append([start, end])

schedule = sorted(schedule)
for i in schedule :
    if not real_schedule or real_schedule[len(real_schedule) - 1][0] != i[0] :
        real_schedule.append(i)

if not real_schedule :
    print(count)
else :
    min_schedules.append(real_schedule[0])
    real_schedule.pop(0)
    for i in real_schedule :
        s, e = i
        s_len = len(min_schedules) - 1

        if min_schedules[s_len][1] > s :
            if min_schedules[s_len][1] > e :
                min_schedules.pop(s_len)
            else :
                continue
        min_schedules.append([s, e])
    print(len(min_schedules) + count)

```

<br>

#### Decompose

``` decompose
def decompose(n) :
    total = n
    n = str(n)

    for i in n :
        total += int(i)

    return total


N = int(input())
initial_num = 0

for i in range(N) :
    if decompose(i) == N :
        initial_num = i
        break

print(initial_num)
```

<br>

#### Bulk

``` bulk
N = int(input())

people = []

for i in range(N) :
    weight, height = map(int, input().split())
    people.append([weight, height, 0])

for person in people :
    count = 1
    for c_person in people :
        if c_person[0] > person[0] and c_person[1] > person[1] :
            count += 1
    person[2] = count

for person in people :
    print(person[2], end=" ")
```

<br>

#### Redraw Chess Board 

``` redraw chess board 
N, M = map(int, input().split())

board, min_count = [], int(1e9)

for _ in range(N) :
    board.append(input())

sample = []
for i in range(8):
    case = []
    for j in range(8):
        if (i + j) % 2 == 0 :
            case.append('B')
        else:
            case.append('W')
    sample.append(case)

for i in range(0, N - 7) :
    for j in range(0, M - 7) :
        count = 0
        for a in range(i, i + 8) :
            for b in range(j, j + 8) :
                if board[a][b] != sample[a - i][b - j] :
                    count += 1

        min_count = min(min_count, count, 64 - count)

print(min_count)
```

<br>

#### Moive Director Syom

``` moive director syom
N = int(input())
index = 665
count = 0
while count != N :
    index += 1
    if "666" in str(index) :
        count += 1

print(index)
```

<br>

#### Make Stack

``` make stack
import sys

N = int(input())
stack = []
for _ in range(N) :
    order = list(map(str, sys.stdin.readline().split()))
    if order[0] == "push" :
        stack.append(order[1])
    elif order[0] == "pop" :
        s_len = len(stack)
        if s_len != 0:
            print(stack.pop(s_len - 1))
        else:
            print(-1)
    elif order[0] == "size" :
        print(len(stack))
    elif order[0] == "empty" :
        if len(stack) == 0:
            print(1)
        else:
            print(0)
    elif order[0] == "top" :
        s_len = len(stack)
        if s_len == 0:
            print(-1)
        else:
            print(stack[s_len - 1])
```

<br>

#### Zero

``` zero
K = int(input())
stack = []

for _ in range(K) :
    data = int(input())
    if data == 0 :
        stack.pop(len(stack) - 1)
    else :
        stack.append(data)

print(sum(stack))
```

<br>

#### VPS

``` VPS
def is_VPS(data) :
    while "()" in data :
        data = data.replace("()", "")

    if data == "" :
        return "YES"
    else:
        return "NO"


T = int(input())

for _ in range(T) :
    print(is_VPS(input()))
```

<br>

#### a Balanced World

```a balanced world
def is_balance(data) :
    while ("()" in data) or ("[]" in data) :
        if "()" in data :
            data = data.replace("()", "")
        if "[]" in data :
            data = data.replace("[]", "")

    if data == "" :
        return "yes"
    else:
        return "no"


while True :
    data = input()
    if data == '.':
        break
    real_data = ""
    for i in data :
        if i == "(" or i == ")" or i == "[" or i == "]" :
            real_data += i

    print(is_balance(real_data))
```

#### Escape from Rectangle

``` Escape from Rectangle
x, y, w, h = map(int, input().split())

print(min(x, w - x, y, h - y))
```

<br>

#### Find the Interval Sum 4

```Find the Interval Sum 4
import sys

input = sys.stdin.readline
N, M = map(int, input().split())
board = list(map(int, input().split()))

start_arr = [0] * (N + 1)
for i in range(N) :
    start_arr[i + 1] = start_arr[i] + board[i]

for _ in range(M) :
    start, end = map(int, input().split())
    print(start_arr[end] - start_arr[start - 1])
```

<br>

#### Sequence

``` sequence
N, K = map(int, input().split())
temperate = list(map(int, input().split()))
t_a = [sum(temperate[:K])]

for i in range(N - K) :
    t_a.append(t_a[i] - temperate[i] + temperate[K + i])

print(max(t_a))
```

<br>

#### human-computer interaction

```human-computer interaction
import sys
input = sys.stdin.readline

S = input().strip()
q = int(input())
a_asc, S_len = ord('a'), len(S)
dp_list = [[0 for _ in range(S_len + 1)] for _ in range(26)]

for i in range(S_len):
    dp_list[ord(S[i]) - a_asc][i + 1] = 1

for i in range(S_len):
    for j in range(26) :
        dp_list[j][i + 1] += dp_list[j][i]

for _ in range(q) :
    a, l, r = input().split()
    print(dp_list[ord(a) - a_asc][int(r) + 1] - dp_list[ord(a) - a_asc][int(l)])
```

<br>

#### Find PrimeNumber

``` find primeNumber
import math


def isPrime(number) :
    if number == 2:
        return True
    elif number == 1 or number % 2 == 0 :
        return False

    for i in range(3, int(math.sqrt(number)) + 1, 2) :
        if number % i == 0 :
            return False
    return True


start, end = map(int, input().split())
for i in range(start, end + 1) :
    if isPrime(i) :
        print(i)
```

<br>

#### Post Bertrand

``` Post Bertrand
import math, sys


def isPrime(number) :
    if number == 2:
        return True
    elif number % 2 == 0 or number == 1 :
        return False

    for i in range(3, int(math.sqrt(number)) + 1, 2) :
        if number % i == 0 :
            return False
    return True


arr = [0] * 246913
for i in range(2, 246913) :
    if isPrime(i) :
        arr[i] = 1

while True :
    data = int(sys.stdin.readline())
    if data == 0 :
        break

    print(sum(arr[data + 1 : data * 2 + 1]))
```

<br>

#### Goldbach's Conjecture

```
import math, sys


def isPrime(number) :
    if number == 2:
        return True
    elif number % 2 == 0 or number == 1 :
        return False

    for i in range(3, int(math.sqrt(number)) + 1, 2) :
        if number % i == 0 :
            return False
    return True


T = int(input())
for _ in range(T) :
    data = int(sys.stdin.readline())
    if data == 4 :
        print("2 2")
    else :
        standard = data // 2
        if standard % 2 == 0 :
            standard -= 1
        for i in range(standard, 1, -2) :
            if isPrime(i) and isPrime(data - i) :
                print(str(i) + " " + str(data - i))
                break
```

<br>

#### Number Card

```
import sys


def binary_search(arr, target, start, end) :
    if start > end :
        return -1
    mid = (start + end) // 2

    if arr[mid] > target :
        return binary_search(arr, target, start, mid - 1)
    elif arr[mid] < target :
        return binary_search(arr, target, mid + 1, end)
    else :
        return mid


N = int(input())
number_cards = list(map(int, sys.stdin.readline().split()))
number_cards = sorted(number_cards)

M = int(input())
sang_cards = list(map(int, sys.stdin.readline().split()))


for card in sang_cards :
    if binary_search(number_cards, card, 0, N - 1) != -1 :
        print("1", end=" ")
    else :
        print("0", end=" ")
```

<br>

#### String Set

``` string set
import sys


def binary_search(arr, target, start, end) :
    if start > end :
        return -1
    mid = (start + end) // 2

    if arr[mid] > target :
        return binary_search(arr, target, start, mid - 1)
    elif arr[mid] < target :
        return binary_search(arr, target, mid + 1, end)
    else :
        return mid


N, M = map(int, input().split())
S = []
for _ in range(N) :
    S.append(sys.stdin.readline())

S_len, total = len(S), 0
S = sorted(S)
for _ in range(M) :
    data = sys.stdin.readline()
    if binary_search(S, data, 0, S_len - 1) != -1 :
        total += 1

print(total)
```

<br>

#### Number Card 2

```
import sys
from bisect import bisect_left, bisect_right


N = int(input())

sang_cards = list(map(int, (sys.stdin.readline()).split()))
sang_cards = sorted(sang_cards)

M = int(input())
confirm_cards = list(map(int, (sys.stdin.readline()).split()))

for card in confirm_cards :
    total = 0
    print(bisect_right(sang_cards, card) - bisect_left(sang_cards, card), end=" ")
```

<br>

#### Make Bridge

``` make bridge
import copy


def find_parent(parent, x) :
    if parent[x] != x :
        return find_parent(parent, parent[x])
    return parent[x]


def union_parent(parent, a, b) :
    a = find_parent(parent, a)
    b = find_parent(parent, b)

    if a > b :
        parent[a] = b
    else:
        parent[b] = a


def dfs(board, x, y, visited, arr) :
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]

    visited[x][y] = 0
    arr.append([x,y])

    for i in range(4) :
        mx, my = x + dx[i], y + dy[i]
        if mx >= 0 and mx < len(board) and my >= 0 and my < len(board[0]) :
            if board[mx][my] == 1 and visited[mx][my] == 1 :
                dfs(board, mx, my, visited, arr)
    return arr


def check_between(islands, island1, island2, point_1, point_2) :
    if point_1[0] == point_2[0] :
        if point_2[1] > point_1[1] :
            b, s = point_2[1], point_1[1]
        else :
            s, b = point_2[1], point_1[1]

        for i in range(s + 1, b) :
            for island in islands :
                if [point_1[0], i] in island :
                    return False
    elif point_1[1] == point_2[1] :
        if point_2[0] > point_1[0] :
            b, s = point_2[0], point_1[0]
        else :
            s, b = point_2[0], point_1[0]

        for i in range(s + 1, b) :
            for island in islands:
                if [i, point_1[1]] in island :
                    return False
    return True


def distances(islands, island1, island2) :
    min_distance = int(1e9)
    for x_1, y_1 in island1 :
        for x_2, y_2 in island2 :
            if x_1 == x_2 :
                d = abs(y_2 - y_1) - 1
                if d >= 2 and check_between(islands, island1, island2, [x_1, y_1], [x_2, y_2]) :
                    min_distance = min(min_distance, d)
            elif y_1 == y_2 :
                d = abs(x_2 - x_1) - 1
                if d >= 2 and check_between(islands, island1, island2, [x_1, y_1], [x_2, y_2]) :
                    min_distance = min(min_distance, d)
    if min_distance == int(1e9) :
        return -1
    return min_distance


N, M = map(int, input().split())

board, visited, islands = [], [], []
for _ in range(N) :
    board.append(list(map(int, input().split())))

visited = copy.deepcopy(board)
for i in range(N) :
    for j in range(M) :
        if visited[i][j] == 1 :
            islands.append(dfs(board, i, j, visited, []))

island_len, island_distance = len(islands), []
for i in range(island_len) :
    for j in range(island_len) :
        if i == j :
            continue
        island_distance.append([distances(islands, islands[i], islands[j]), i + 1, j + 1])

island_distance = sorted(island_distance, key=lambda x : x[0])
total, parent = 0, [0] * (island_len + 1)
for i in range(1, island_len + 1) :
    parent[i] = i

for cost, a, b in island_distance :
    if find_parent(parent, a) != find_parent(parent, b) and cost >= 2 :
        total += cost
        union_parent(parent, a, b)

standard, parent_state = find_parent(parent, 1), True
for i in range(2, island_len + 1) :
    if standard != find_parent(parent, i) :
        parent_state = False
        break

if not parent_state :
    print(-1)
else :
    print(total)
```

<br>

#### ACM craft

``` ACM Craft
import heapq, sys

T = int(input())

for _ in range(T) :
    N, K = map(int, input().split())
    D_list = list(map(str, sys.stdin.readline().split()))

    edges, indeg, min_time = [[] for _ in range(N + 1)], [0] * (N + 1), 0
    for i in range(K) :
        x, y = map(int, (sys.stdin.readline()).split())
        edges[x].append(y)
        indeg[y] += 1

    W = int(input())
    q = []
    for i in range(1, N + 1) :
        if indeg[i] == 0 :
            heapq.heappush(q, [int(D_list[i - 1]), i])
    while q :
        time, node = heapq.heappop(q)
        min_time = time
        if node == W :
            break
        for edge in edges[node] :
            indeg[edge] -= 1
            if indeg[edge] == 0 :
                heapq.heappush(q, [int(D_list[edge - 1]) + int(min_time), edge])

    print(min_time)
```

<br>

#### Book Page

``` book page
import copy


def cal_1(len_N) :
    data = [0] * 10
    for i in range(len_N, 0, -1):
        zero_a = int(9 * (len_N - i - 1) * (10 ** (len_N - i - 2)))
        data[0] += zero_a
        for j in range(1, 10):
            data[j] += int(zero_a + 10 ** (len_N - i - 1))
    return data


def cal_2(N) :
    list_N_2 = list(str(N))
    len_N_2 = len(list_N_2)
    stand = cal_1(len_N_2)
    total_2 = copy.deepcopy(stand)

    for i in range(1, int(list_N_2[0])):
        total_2[i] += 10 ** (len_N_2 - 1) - 1

        for j in list(str(i * (10 ** (len_N_2 - 1)))):
            total_2[int(j)] += 1

        for j in range(10):
            total_2[j] += stand[j]

        total_2[0] += ((len_N_2 - 1) * (10 ** (len_N_2 - 1) - 1) - sum(stand))

    for i in list(str(int(list_N_2[0]) * (10 ** (len_N_2 - 1)))):
        total_2[int(i)] += 1

    total_2[int(list_N_2[0])] += int("".join(list_N_2[1:]))

    return int("".join(list_N_2[1:])), total_2


N = int(input())
N_list = list(str(N))
N_len = len(N_list)

if N >= 10 :
    total_number = N_len * (N - (10 ** (N_len - 1) - 1))
    for i in range(1, N_len) :
        total_number += (N_len - i) * (10 ** (N_len - i - 1)) * 9
else :
    total_number = int(N_list[N_len - 1])

real_total = [0] * 10
while N >= 10 :
    N, total = cal_2(N)
    for i in range(10):
        real_total[i] += total[i]

for i in range(1, N + 1) :
    real_total[i] += 1

real_total[0] += total_number - sum(real_total)
for i in real_total :
    print(i, end=" ")

arr = [0] * 10
for i in range(1, 11) :
    for j in list(str(i)) :
        arr[int(j)] += 1

print(arr)
```

<br>

#### Prime Twice

``` prime twice
import copy


def makePrime(number) :
    states = [True] * number
    for i in range(2, int(number ** 0.5)) :
        if states[i] :
            for j in range(i + i, number, i) :
                states[j] = False
    return [i for i in range(2, number) if states[i]]


def dfs(number) :
    if visited[number] :
        return False
    visited[number] = True

    for i in odd :
        if copy_list[i] + copy_list[even[number - 1]] in primeNumbers :
            if matched[i] == 0 or dfs(matched[i]) :
                matched[i] = number
                return True
    return False


N = int(input())
numbers = list(map(int, input().split()))
first_number, primeNumbers = numbers[0], makePrime(2000)
result_arr, even_odd = [], True

for i in range(1, N) :
    if first_number + numbers[i] in primeNumbers :
        result, state = [], True
        copy_list = copy.deepcopy(numbers)
        copy_list.pop(i)
        copy_list.pop(0)

        matched = [0 for _ in range(len(copy_list) + 1)]
        odd, even = [], []
        for j in range(len(copy_list)) :
            if copy_list[j] % 2 == 0 :
                even.append(j)
            else :
                odd.append(j)

        if len(even) != len(odd) :
            even_odd = False
            break

        for j in range(1, N // 2) :
            visited = [False] * (len(copy_list) + 1)
            dfs(j)

        total = 0
        for j in matched :
            if j != 0 :
                total += 1
        if total == (N // 2) - 1 :
            result_arr.append(numbers[i])


if not result_arr or not even_odd:
    print(-1)
else :
    result_arr = sorted(result_arr)
    for i in result_arr :
        print(i, end=" ")
```

<br>

#### Fly me to the Alpha Centauri

``` Fly me to the Alpha Centauri
T = int(input())

for _ in range(T) :
    x, y = map(int, input().split())
    cur_location, cur_speed, total_distance = 0, 0, y - x
    result = 0

    if y - x == 1 :
        print(1)
    else :
        while total_distance >= (cur_speed * (cur_speed + 1)) // 2 + (cur_speed * (cur_speed - 1)) // 2 :
            cur_speed += 1

        cur_speed -= 1
        result = (cur_speed - 1) * 2 + 1
        cur_location = total_distance - (cur_speed * (cur_speed + 1)) // 2 - (cur_speed * (cur_speed - 1)) // 2

        for i in range(cur_speed, 0, -1) :
            while cur_location >= i :
                cur_location -= i
                result += 1

        print(result)
```

<br>

#### Break the Wall and Move

``` break the wall and move
import copy
from collections import deque

N, M = map(int, input().split())
board = []
for _ in range(N) :
    board.append(list(input()))

dx, dy = [1, 0, -1, 0], [0, 1, 0, -1]
q, visited = deque(), [[False for _ in range(M)] for _ in range(N)]
q.append([0 ,0, 1, 1])
visited[0][0], result = True, -1
b_visited = copy.deepcopy(visited)

while q :
    x, y, b, dist = q.popleft()

    if x == N - 1 and y == M - 1 :
        result = dist
        break

    for i in range(4) :
        mx, my = x + dx[i], y + dy[i]
        if mx >= 0 and mx < N and my >= 0 and my < M and not visited[mx][my] :
            if b == 0 and b_visited[mx][my] :
                continue
            if board[mx][my] == '0' :
                q.append([mx, my, b, dist + 1])
                b_visited[mx][my] = True
                if b == 1 :
                    visited[mx][my] = True
            elif board[mx][my] == '1' and b == 1 :
                q.append([mx, my, b - 1, dist + 1])
                b_visited[mx][my] = True
                visited[mx][my] = True

print(result)
```

<br>

#### Remote

``` remote
N = int(input())
M = int(input())
r, dp = [], [int(1e9)] * 1000000
if M != 0 :
    r = list(map(str, input().split()))

if not r :
    if abs(N - 100) > 2 :
        print(len(str(N)))
    else :
        print(abs(N - 100))

else :
    for i in range(1000000) :
        state = True
        for j in str(i) :
            if j in r :
                state = False
                break

        if state :
            dp[i] = len(str(i))

    dp[100] = 0
    for i in range(1, 1000000) :
        dp[i] = min(dp[i - 1] + 1, dp[i])

    for i in range(999999, -1, -1) :
        dp[i - 1] = min(dp[i] + 1, dp[i - 1])

    print(dp[N])
```

<br>

#### Knapsack Algorithm

``` Knapsack Algorithm
N, K = map(int, input().split())

db, dp = [[0,0]], [[0 for _ in range(K + 1)] for _ in range(N + 1)]
for _ in range(N) :
    db.append(list(map(int, input().split())))

for i in range(1, N + 1) :
    for j in range(1, K + 1) :
        weight, value = db[i][0], db[i][1]

        if j < weight :
            dp[i][j] = dp[i - 1][j]
        else :
            dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight] + value)

print(dp[N][K])
```

<br>

#### tetromino

``` tetromino
import sys
from itertools import combinations

max_total = 0
sp_case = [[1, 0], [-1, 0], [0, 1], [0, -1]]


def dfs(x, y, visited, result, dist) :
    global max_total, max_board

    if max_total >= result + max_board * (4 - dist) :
        return

    for k in range(4) :
        mx, my = x + dx[k], y + dy[k]
        if mx >= 0 and mx < N and my >= 0 and my < M and visited[mx][my] :
            if dist == 3 :
                max_total = max(max_total, result + board[mx][my])
                continue
            visited[mx][my] = False
            dfs(mx, my, visited, result + board[mx][my], dist + 1)
            visited[mx][my] = True


N, M = map(int, input().split())
board, max_data = [], []
dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]

for _ in range(N) :
    data = list(map(int, (sys.stdin.readline()).split()))
    max_data.append(max(data))
    board.append(data)

visited, max_board = [[True for _ in range(M)] for _ in range(N)], max(max_data)

for i in range(N) :
    for j in range(M) :
        visited[i][j], result = False, board[i][j]
        dfs(i,j, visited, result, 1)
        visited[i][j] = True

        for case in combinations(sp_case, 3) :
            sp_sum = board[i][j]
            for s_x, s_y in case :
                mx, my = i + s_x, j + s_y
                if mx >= 0 and mx < N and my >= 0 and my < M :
                    sp_sum += board[mx][my]
                else :
                    break
            max_total = max(max_total, sp_sum)

print(max_total)
```

<br>

#### Downhill

``` downhill
import sys
from collections import deque

N, M = map(int, input().split())

board, total = [], 1
for _ in range(N) :
    board.append(list(map(int, (sys.stdin.readline()).split())))

dx, dy = [1, -1, 0, 0], [0, 0, -1, 1]
dp = [[0 for _ in range(M)] for _ in range(N)]

q, dp[0][0] = deque(), 1
q.append([0, 0])

while q :
    x, y = q.popleft()

    if x == N - 1 and y == M - 1:
        continue

    for i in range(4) :
        mx, my = x + dx[i], y + dy[i]
        if mx >= 0 and mx < N and my >= 0 and my < M and board[x][y] > board[mx][my] :
            dp[mx][my] += dp[x][y]

            if [mx, my] not in q :
                q.append([mx, my])
    dp[x][y] = 0

print(dp[N - 1][M - 1])
```

<br>

#### ADD 1, 2, 3

``` add 1, 2, 3
import copy

dp = [[] for _ in range(12)]
dp[1].append([1])
dp[2].append([2])
dp[3].append([3])
q, temp = [[1], [2], [3]], []

for _ in range(11) :
    for i in q :
        for j in range(1, 4) :
            c = sum(i) + j
            if 12 > c :
                i_copy = copy.deepcopy(i)
                i_copy.append(j)
                if i_copy not in dp[c] :
                    dp[c].append(i_copy)
                    temp.append(i_copy)
    q = copy.deepcopy(temp)

T = int(input())
for _ in range(T) :
    print(len(dp[int(input())]))
```

<br>

#### 2048

``` 2048
import copy


def move(state, real_board, dist) :
    global max_value
    board = copy.deepcopy(real_board)

    if state == "up" :
        for i in range(N) :
            for j in range(N - 1) :
                if board[j][i] == 0 :
                    for k in range(j + 1, N) :
                        if board[k][i] != 0 :
                            board[j][i] = board[k][i]
                            board[k][i] = 0
                            break
                    if board[j][i] == 0 :
                        break
            for j in range(N - 1) :
                if board[j][i] == 0 or board[j + 1][i] == 0 :
                    continue
                if board[j][i] == board[j + 1][i] :
                    board[j][i] *= 2
                    board[j + 1][i] = 0
            for j in range(N - 1) :
                if board[j][i] == 0 :
                    for k in range(j + 1, N) :
                        if board[k][i] != 0 :
                            board[j][i] = board[k][i]
                            board[k][i] = 0
                            break
                    if board[j][i] == 0 :
                        break
    elif state == "down" :
        for i in range(N - 1, -1, -1) :
            for j in range(N - 1, 0, -1) :
                if board[j][i] == 0 :
                    for k in range(j - 1, -1, -1) :
                        if board[k][i] != 0 :
                            board[j][i] = board[k][i]
                            board[k][i] = 0
                            break
                    if board[j][i] == 0 :
                        break
            for j in range(N - 1, 0, -1) :
                if board[j][i] == 0 or board[j - 1][i] == 0 :
                    continue
                if board[j][i] == board[j - 1][i] :
                    board[j][i] *= 2
                    board[j - 1][i] = 0
            for j in range(N - 1, 0, -1) :
                if board[j][i] == 0 :
                    for k in range(j - 1, -1, -1) :
                        if board[k][i] != 0 :
                            board[j][i] = board[k][i]
                            board[k][i] = 0
                            break
                    if board[j][i] == 0 :
                        break
    elif state == "left" :
        for j in range(N) :
            for i in range(N - 1) :
                if board[j][i] == 0 :
                    for k in range(i + 1, N) :
                        if board[j][k] != 0 :
                            board[j][i] = board[j][k]
                            board[j][k] = 0
                            break
                    if board[j][i] == 0 :
                        break
            for i in range(N - 1) :
                if board[j][i] == 0 or board[j][i + 1] == 0 :
                    continue
                if board[j][i] == board[j][i + 1] :
                    board[j][i] *= 2
                    board[j][i + 1] = 0
            for i in range(N - 1) :
                if board[j][i] == 0 :
                    for k in range(i + 1, N) :
                        if board[j][k] != 0 :
                            board[j][i] = board[j][k]
                            board[j][k] = 0
                            break
                    if board[j][i] == 0 :
                        break
    elif state == "right" :
        for j in range(N - 1, -1, -1):
            for i in range(N - 1, 0, -1) :
                if board[j][i] == 0 :
                    for k in range(i - 1, -1, -1) :
                        if board[j][k] != 0 :
                            board[j][i] = board[j][k]
                            board[j][k] = 0
                            break
                    if board[j][i] == 0 :
                        break
            for i in range(N - 1, 0, -1):
                if board[j][i] == 0 or board[j][i - 1] == 0:
                    continue
                if board[j][i] == board[j][i - 1]:
                    board[j][i] *= 2
                    board[j][i - 1] = 0
            for i in range(N - 1, 0, -1) :
                if board[j][i] == 0 :
                    for k in range(i - 1, -1, -1) :
                        if board[j][k] != 0 :
                            board[j][i] = board[j][k]
                            board[j][k] = 0
                            break
                    if board[j][i] == 0 :
                        break

    if dist == 4 :
        for i in range(N) :
            max_value = max(max_value, max(board[i]))
        return

    move("up", board, dist + 1)
    move("down", board, dist + 1)
    move("left", board, dist + 1)
    move("right", board, dist + 1)


N = int(input())

o_board, max_value = [], 0
for _ in range(N) :
    o_board.append(list(map(int, input().split())))

move("up", o_board, 0)
move("down", o_board, 0)
move("left", o_board, 0)
move("right", o_board, 0)

print(max_value)
```

<br>

#### Math Word

``` math word
from collections import defaultdict

N = int(input())
words, v = [], defaultdict(int)
for _ in range(N) :
    words.append(input())

for word in words :
    s = len(word) - 1
    for a in word :
        v[a] += 10 ** s
        s -= 1

v = sorted(v, key=lambda x : v[x], reverse=True)

sel, total = 9, 0
for a in v :
    for i in range(N):
        words[i] = words[i].replace(a, str(sel))
    sel -= 1

for word in words :
    total += int(word)
print(total)
```

<br>

#### repairman sailing

``` repairman sailing
import sys

N, L = map(int, input().split())
w_list = list(map(int, (sys.stdin.readline()).split()))
w_list = sorted(w_list)
index, total = 0, 0

while N > index :
    e = w_list[index] + L - 1
    while N > index and e >= w_list[index] :
        index += 1

    total += 1

print(total)
```

<br>

#### two solutions

``` two solutions
N = int(input())
N_list = list(map(int, input().split()))

A = []
for i in N_list :
    if i > 0 :
        A.append([i, 0])
    else :
        A.append([-i, 1])

A = sorted(A, key=lambda x : x[0])

min_sel, s, e = int(1e11), -1, -1
for i in range(N - 1) :
    if A[i + 1][1] != A[i][1] :
        cost = A[i + 1][0] - A[i][0]
    else :
        cost = A[i + 1][0] + A[i][0]

    if min_sel > abs(cost) :
        min_sel, s, e = abs(cost), i, i + 1

result = []

if A[s][1] == 1 :
    result.append(-A[s][0])
else :
    result.append(A[s][0])

if A[e][1] == 1 :
    result.append(-A[e][0])
else:
    result.append(A[e][0])

result = sorted(result)
print(str(result[0]) + " " + str(result[1]))
```

<br>

#### succession of prime numbers

```succession of prime numbers
def binary_search(arr, target, start, end, standard) :
    if start > end :
        return -1
    mid = (start + end) // 2

    if standard - arr[mid] < target :
        return binary_search(arr, target, start, mid - 1, standard)
    elif standard - arr[mid] > target :
        return binary_search(arr, target, mid + 1, end, standard)
    else :
        return mid


N = int(input())

if N == 1 :
    print(0)
else :
    total = 0
    state = [True for _ in range(N + 1)]
    for i in range(2, int((N + 1) ** 0.5) + 1):
        if state[i]:
            for j in range(i + i, N + 1, i):
                state[j] = False
    primeList = [i for i in range(2, N + 1) if state[i]]

    continuous = [0]
    for i in range(len(primeList)):
        continuous.append(primeList[i] + continuous[i])

    continuous_len = len(continuous)
    for i in range(1, len(continuous)):
        if N > continuous[i] :
            continue

        if binary_search(continuous, N, 0, continuous_len - 1, continuous[i]) != -1 :
            total += 1

    print(total)
```

<br>

#### subtotal

```subtotal
import sys


def binary_search(arr, target, start, end, standard) :
    global state
    if start > end :
        return -1
    mid = (start + end) // 2

    if standard - arr[mid] >= target :
        state = mid
        return binary_search(arr, target, mid + 1, end, standard)
    elif standard - arr[mid] < target :
        return binary_search(arr, target, start, mid - 1, standard)


N, S = map(int, input().split())
N_list = list(map(int, (sys.stdin.readline()).split()))
continuous, min_sel = [0], int(1e9)
state = -1

for i in range(N) :
    continuous.append(N_list[i] + continuous[i])

if S > continuous[N] :
    print(0)
else :
    for i in range(N + 1) :
        state = -1
        if S > continuous[i] :
            continue

        binary_search(continuous, S, 0, N, continuous[i])
        if state != -1 :
            min_sel = min(min_sel, i - state)

    print(min_sel)
```

<br>

#### LCS

``` LCS
str_1 = input()
str_2 = input()

str_len_1 = len(str_1)
dp = [0] * str_len_1
for i in str_2 :
    pre_sel = 0
    for j in range(str_len_1) :
        if dp[j] > pre_sel:
            pre_sel = dp[j]
        elif str_1[j] == i :
            dp[j] = pre_sel + 1

print(max(dp))
```

<br>

#### population movement

``` population movement
import sys
from collections import deque

N, L, R = map(int, sys.stdin.readline().split())
board, result = [], 0

for _ in range(N) :
    board.append(list(map(int, (sys.stdin.readline()).split())))

dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]

while True :
    visited = [[True for _ in range(N)] for _ in range(N)]
    state, q = True, deque()
    for i in range(N) :
        for j in range(N) :
            if visited[i][j] :
                total_p, total_c = 0, []
                q.append([i, j])
                visited[i][j] = False

                while q :
                    x, y = q.popleft()
                    if [x, y] not in total_c :
                        total_p += board[x][y]
                        total_c.append([x, y])

                    for k in range(4) :
                        mx, my = x + dx[k], y + dy[k]
                        if 0 <= mx < N and 0 <= my < N and visited[mx][my] :
                            if L <= abs(board[x][y] - board[mx][my]) <= R :
                                q.append([mx, my])
                                visited[mx][my] = False

                t_c = len(total_c)
                if t_c > 1 :
                    state = False
                    average_p = int(total_p / t_c)
                    for x, y in total_c:
                        board[x][y] = average_p

    if state :
        break
    result += 1


print(result)
```

<br>

#### specific shortest path

```specific shortest path
import heapq
import sys


def d(start, end) :
    distance = [int(1e9) for _ in range(N + 1)]
    q, distance[start] = [], 0
    heapq.heappush(q, [0, start])

    while q :
        dist, node = heapq.heappop(q)

        if dist > distance[node] :
            continue

        for i in graph[node] :
            cost = dist + i[0]
            if distance[i[1]] > cost :
                heapq.heappush(q, [cost, i[1]])
                distance[i[1]] = cost

    return distance[end]


N, E = map(int, (sys.stdin.readline()).split())

graph = [[] for _ in range(N + 1)]
for _ in range(E) :
    a, b, c = map(int, (sys.stdin.readline()).split())
    graph[a].append([c, b])
    graph[b].append([c, a])

v1, v2 = map(int, (sys.stdin.readline()).split())

result = min(d(1, v1) + d(v1, v2) + d(v2, N), d(1, v2) + d(v2, v1) + d(v1, N))
if result >= int(1e9) :
    print(-1)
else :
    print(result)
```

<br>

#### coin 1

``` coin 1
import sys

n, k = map(int, input().split())
coins = []
for _ in range(n) :
    coin = int(sys.stdin.readline().strip())
    if 10001 > coin :
        coins.append(coin)

coins = sorted(coins)
dp = [0] * 10001

for coin in coins :
    dp[coin] += 1
    for i in range(1, 10001 - coin) :
        if dp[i] != 0 :
            dp[i + coin] += dp[i]

print(dp[k])
```

<br>

#### chicken delivery

``` chicken delivery
import sys
from itertools import combinations


def cal_distance(p1, p2) :
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


N, M = map(int, (sys.stdin.readline()).split())
board, chi, house, min_distance = [], [], [], int(1e9)

for i in range(N) :
    data = list(map(int, (sys.stdin.readline()).split()))
    for j in range(N) :
        if data[j] == 2 :
            chi.append([i, j])
        elif data[j] == 1 :
            house.append([i, j])
    board.append(data)

for combin in combinations(chi, M) :
    total_dist = 0
    for h1 in house :
        house_chi = int(1e9)
        for c1 in combin :
            house_chi = min(house_chi, cal_distance(h1, c1))
        total_dist += house_chi

    min_distance = min(min_distance, total_dist)
print(min_distance)
```

<br>

### AC

``` AC
import sys
from collections import deque

T = int(input())
for _ in range(T) :
    state_error, state_reverse = False, False
    p = sys.stdin.readline().strip()
    n = int(sys.stdin.readline())
    arr = input()
    if arr == "[]" :
        arr = deque()
    else :
        arr = deque(list(map(int, arr[1:-1].split(","))))

    for i in p :
        if arr :
            if i == "R" :
                if state_reverse :
                    state_reverse = False
                else :
                    state_reverse = True
            elif i == "D" :
                if state_reverse :
                    arr.pop()
                else :
                    arr.popleft()
        else :
            if i == "D":
                state_error = True
                print("error")
                break

    if not state_error :
        if state_reverse :
            arr.reverse()

        arr_len = len(arr)
        print("[", end="")
        for i in range(arr_len):
            if i != arr_len - 1:
                print(arr[i], end=",")
            else:
                print(arr[i], end="")
        print("]")
```

<br>

#### Sudoku

```Sudoku
import sys
from collections import deque

board, empty_list = [], deque()
for i in range(9) :
    data = list(map(int, (sys.stdin.readline()).split()))
    for j in range(9):
        if data[j] == 0 :
            empty_list.append([i, j])
    board.append(data)


def find(f_board) :
    if not empty_list :
        for i in range(9) :
            for j in range(9) :
                print(f_board[i][j], end=" ")
            print()
        return "True"

    i, j = empty_list.popleft()
    candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for s in range(9):
        if f_board[s][j] in candidate:
            candidate.remove(f_board[s][j])
        if f_board[i][s] in candidate:
            candidate.remove(f_board[i][s])

    if not candidate:
        empty_list.appendleft([i, j])
        return -1

    standard_i, standard_j = (i // 3) * 3, (j // 3) * 3
    for x in range(standard_i, standard_i + 3):
        for y in range(standard_j, standard_j + 3):
            if f_board[x][y] in candidate:
                candidate.remove(f_board[x][y])

    if not candidate:
        empty_list.appendleft([i, j])
        return -1

    for c in candidate:
        temp = f_board[i][j]
        f_board[i][j] = c
        if find(f_board) == "True":
            return "True"
        else:
            f_board[i][j] = temp

    empty_list.appendleft([i, j])
    return -1


find(board)
```

<br>

#### snack

``` snack
import sys
from collections import deque

N = int(input())
K = int(input())
apples = []
for _ in range(K) :
    apples.append(list(map(int, (sys.stdin.readline()).split())))

L = int(input())
dx, dy = [0, 1, 0, -1], [1, 0, -1, 0]
move, data, current_location, current_time = deque(), [], 0, 0
cx, cy, snack_body = 1, 1, deque()
snack_body.append([1, 1])
for _ in range(L) :
    data.append(list(map(str, sys.stdin.readline().split())))

move.append([int(data[0][0]), "N"])
for i in range(L - 1) :
    move.append([int(data[i + 1][0]) - int(data[i][0]), data[i][1]])
move.append([101, data[L - 1][1]])

state = True
while move and state:
    second, location = move.popleft()
    if location == "D" :
        current_location = (current_location + 1) % 4
    elif location == "L" :
        current_location = (current_location + 3) % 4

    for _ in range(int(second)) :
        mx, my = cx + dx[current_location], cy + dy[current_location]
        current_time += 1
        if 1 <= mx < N + 1 and 1 <= my < N + 1 and [mx, my] not in snack_body :
            if [mx, my] in apples :
                apples.remove([mx, my])
            else:
                snack_body.popleft()
            snack_body.append([mx, my])
            cx, cy = mx, my
        else :
            state = False
            break

print(current_time)
```

<br>

#### interleave operator

``` interleave operator
import copy, sys


def cal(current, dist, c_op) :
    global max_value, min_value
    if dist == N:
        max_value = max(max_value, current)
        min_value = min(min_value, current)
        return

    for o in range(4) :
        if o in c_op :
            copy_op = copy.deepcopy(c_op)
            copy_op.remove(o)
            if o != 3 or current >= 0:
                cal(eval(str(current) + oper[o] + str(A_list[dist])), dist + 1, copy_op)
            else:
                cal(((current * -1) // A_list[dist]) * -1, dist + 1, copy_op)


N = int(input())
A_list = list(map(int, (sys.stdin.readline()).split()))
op_list = list(map(int, (sys.stdin.readline()).split()))
op = []
for i in range(4) :
    for j in range(op_list[i]) :
        op.append(i)
oper = ['+', '-', '*', '//']
max_value, min_value = -int(1e10), int(1e10)

cal(A_list[0], 1, op)
print(max_value)
print(min_value)
```

<br>

#### move pipe 1

``` move pipe 1
import sys

N = int(sys.stdin.readline())
total, board = 0, []
for _ in range(N) :
    board.append(list(map(int, (sys.stdin.readline()).split())))

if board[N - 1][N - 1] == 1 or (board[N - 2][N - 1] == 1 and board[N - 1][N - 2] == 1):
    print(0)
else :
    current_ap = 0
    dx, dy = [[0, 1], [1, 1], [0, 1, 1]], [[1, 1], [0, 1], [1, 0, 1]]


    def dfs(x, y, ap) :
        if x == N - 1 and y == N - 1 :
            global total
            total += 1
            return

        if (ap == 0 and y == N - 1) or (ap == 1 and x == N - 1):
            return

        if ap == 2 :
            for i in range(3) :
                mx, my = x + dx[ap][i], y + dy[ap][i]
                if mx < N and my < N and board[mx][my] == 0:
                    if i == 2:
                        if not (board[mx - 1][my] == 0 and board[mx][my - 1] == 0):
                            continue
                    dfs(mx, my, i)
        else :
            for i in range(2) :
                mx, my = x + dx[ap][i], y + dy[ap][i]
                if mx < N and my < N and board[mx][my] == 0:
                    if i == 1:
                        if not (board[mx - 1][my] == 0 and board[mx][my - 1] == 0):
                            continue
                    if i == 0 :
                        dfs(mx, my, ap)
                    else :
                        dfs(mx, my, 2)


    dfs(0, 1, 0)
    print(total)
```

<br>

#### War

``` war
import sys
from collections import deque

M, N = map(int, (sys.stdin.readline()).split())
board = []
for _ in range(N) :
    board.append(list(sys.stdin.readline().strip()))

dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]
visited = [[True for _ in range(M)] for _ in range(N)]
a_total, b_total = 0, 0

for i in range(N) :
    for j in range(M) :
        if visited[i][j] :
            visited[i][j], total = False, 0
            q = deque()
            q.append([i, j])
            while q :
                x, y = q.popleft()
                total += 1

                for k in range(4) :
                    mx, my = x + dx[k], y + dy[k]
                    if 0 <= mx < N and 0 <= my < M and visited[mx][my] and board[i][j] == board[mx][my]:
                        visited[mx][my] = False
                        q.append([mx, my])

            if board[i][j] == 'B' :
                b_total += total ** 2
            else :
                a_total += total ** 2

print(str(a_total) + " " + str(b_total))
```

<br>

#### hide and seek 2

``` hide and seek 2
import sys


standard = 100000
N, K = map(int, (sys.stdin.readline()).split())
if K > N :
    dp = [[0, 0] for _ in range(standard + 1)]

    for i in range(N, 0, -1) :
        dp[i - 1][0] = dp[i][0] + 1

    for i in range(N, standard) :
        dp[i + 1][0] = dp[i][0] + 1

    for i in range(standard + 1) :
        if standard + 1 > i + 1 :
            if dp[i + 1][0] > dp[i][0]:
                if dp[i + 1][0] > dp[i][0] + 1:
                    dp[i + 1][0] = dp[i][0] + 1
                    dp[i + 1][1] = max(dp[i][1], 1)
                elif dp[i + 1][0] == dp[i][0] + 1:
                    dp[i + 1][1] += max(dp[i][1], 1)
            elif dp[i + 1][0] < dp[i][0]:
                if dp[i][0] > dp[i + 1][0] + 1:
                    dp[i][0] = dp[i + 1][0] + 1
                    dp[i][1] = max(dp[i + 1][1], 1)
                elif dp[i][0] == dp[i + 1][0] + 1:
                    dp[i][1] += max(dp[i + 1][1], 1)
        if standard + 1 > i * 2:
            if dp[i * 2][0] > dp[i][0]:
                if dp[i * 2][0] > dp[i][0] + 1:
                    dp[i * 2][0] = dp[i][0] + 1
                    dp[i * 2][1] = max(dp[i][1], 1)
                elif dp[i * 2][0] == dp[i][0] + 1:
                    dp[i * 2][1] += max(dp[i][1], 1)
            elif dp[i * 2][0] < dp[i][0]:
                if dp[i][0] > dp[i * 2][0] + 1:
                    dp[i][0] = dp[i * 2][0] + 1
                    dp[i][1] = max(dp[i * 2][1], 1)
                elif dp[i][0] == dp[i * 2][0] + 1:
                    dp[i][1] += max(dp[i * 2][1], 1)
        # print(str(i), end=" ")
        # print(dp)
    print(dp[K][0])
    print(dp[K][1])
    # print(dp)
else :
    print(N - K)
    print(1)
```

<br>

#### hide and seek 4

``` hide and seek 4
import copy
import sys


standard = 100000
N, K = map(int, (sys.stdin.readline()).split())
if K > N :
    dp = [[0, [i]] for i in range(standard + 1)]

    for i in range(N, 0, -1) :
        dp[i - 1][0] = dp[i][0] + 1

    for i in range(N, standard) :
        dp[i + 1][0] = dp[i][0] + 1

    for i in range(standard + 1) :
        if standard + 1 > i + 1 :
            if dp[i + 1][0] > dp[i][0]:
                if dp[i + 1][0] > dp[i][0] + 1:
                    dp[i + 1][0] = dp[i][0] + 1
                    dp[i + 1][1] = copy.deepcopy(dp[i][1])
                    dp[i + 1][1].append(i + 1)
            elif dp[i + 1][0] < dp[i][0]:
                if dp[i][0] > dp[i + 1][0] + 1:
                    dp[i][0] = dp[i + 1][0] + 1
                    dp[i][1] = copy.deepcopy(dp[i + 1][1])
                    dp[i][1].append(i)
        if standard + 1 > i * 2:
            if dp[i * 2][0] > dp[i][0]:
                if dp[i * 2][0] > dp[i][0] + 1:
                    dp[i * 2][0] = dp[i][0] + 1
                    dp[i * 2][1] = copy.deepcopy(dp[i][1])
                    dp[i * 2][1].append(i * 2)
            elif dp[i * 2][0] < dp[i][0]:
                if dp[i][0] > dp[i * 2][0] + 1:
                    dp[i][0] = dp[i * 2][0] + 1
                    dp[i][1] = copy.deepcopy(dp[i * 2][1])
                    dp[i][1].append(i)

    print(dp[K][0])
    standard = dp[K][1][0]
    if N > standard :
        for i in range(N, standard, -1) :
            print(i, end=" ")
    elif N < standard :
        for i in range(N, standard, 1) :
            print(i, end=" ")

    for i in dp[K][1] :
        print(i, end=" ")
elif K == N :
    print(0)
    print(str(N))
else :
    print(N - K)
    for i in range(N, K - 1, -1) :
        print(i, end=" ")
```

<br>

#### hide and seek 3

``` hide and seek 3
import sys


standard = 100000
N, K = map(int, (sys.stdin.readline()).split())
if K > N :
    dp = [0 for _ in range(standard + 1)]

    for i in range(N, 0, -1) :
        dp[i - 1] = dp[i] + 1

    for i in range(N, standard) :
        dp[i + 1] = dp[i] + 1

    for i in range(standard + 1) :
        if standard + 1 > i + 1 :
            if dp[i + 1] > dp[i]:
                if dp[i + 1] > dp[i] + 1:
                    dp[i + 1] = dp[i] + 1
            elif dp[i + 1] < dp[i]:
                if dp[i] > dp[i + 1] + 1:
                    dp[i] = dp[i + 1] + 1
        if standard + 1 > i * 2:
            if dp[i * 2] > dp[i]:
                if dp[i * 2] > dp[i]:
                    dp[i * 2] = dp[i]
            elif dp[i * 2] < dp[i]:
                if dp[i] > dp[i * 2]:
                    dp[i] = dp[i * 2]
    print(dp[K])
else :
    print(N - K)
```

<br>

#### emoticon

``` emoticon
S = int(input())
dp = [int(1e9)] * 1001
dp[0], dp[1] = 1, 0
for i in range(1, 1001) :
    index = 1
    for j in range(i + i, 1001, i) :
        dp[j] = min(dp[j], dp[i] + index + 1)
        index += 1

        sel = 1
        for k in range(j - 1, 0, -1) :
            dp[k] = min(dp[k], dp[j] + sel)
            sel += 1

print(dp[S])
```

<br>

#### jump

``` jump
import sys

N = int(sys.stdin.readline().strip())

board, dp = [], [[0 for _ in range(N)] for _ in range(N)]
for _ in range(N) :
    board.append(list(map(int, (sys.stdin.readline()).split())))

dp[0][0] = 1
q = [[] for _ in range(N * 2 - 2)]
q[0].append([0, 0, board[0][0]])

for i in range(N * 2 - 2) :
    for x, y, jump in q[i] :
        if N > x + jump :
            jump_x = board[x + jump][y]
            if (x + jump == N - 1 and y == N - 1) or jump_x != 0 :
                dp[x + jump][y] += dp[x][y]
                if N * 2 - 2 > x + y + jump and [x + jump, y, jump_x] not in q[x + y + jump]:
                    q[x + y + jump].append([x + jump, y, jump_x])

        if N > y + jump :
            jump_y = board[x][y + jump]
            if (x == N - 1 and y + jump == N - 1) or jump_y != 0 :
                dp[x][y + jump] += dp[x][y]
                if N * 2 - 2 > x + y + jump and [x, y + jump, jump_y] not in q[x + y + jump]:
                    q[x + y + jump].append([x, y + jump, jump_y])

print(dp[N - 1][N - 1])
```

<br>

#### 1, 2, 3 더하기 4

```1, 2, 3 더하기 4
import sys

T = int(input())
dp = [0] * 10001

for number in range(1, 4) :
    dp[number] += 1
    for i in range(1, 10001) :
        if 10001 > i + number :
            dp[i + number] += dp[i]

for _ in range(T) :
    print(dp[int(sys.stdin.readline().strip())])
```

<br>

#### Run

``` run
import sys
from collections import deque

N, M, K = map(int, (sys.stdin.readline()).split())

board = []
for _ in range(N) :
    board.append(list(sys.stdin.readline().strip()))

x1, y1, x2, y2 = map(int, (sys.stdin.readline()).split())
x1, y1, x2, y2 = x1 - 1, y1 - 1, x2 - 1, y2 - 1
dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]

dp = [[int(1e9) for _ in range(M)] for _ in range(N)]
q, state, dp[x1][y1] = deque(), True, 0
q.append([x1, y1])

while q and state:
    x, y = q.popleft()

    for i in range(4) :
        mx, my, k = x + dx[i], y + dy[i], 1
        while 0 <= mx < N and 0 <= my < M and board[mx][my] == '.' and K >= k and dp[mx][my] >= dp[x][y] + 1:
            if mx == x2 and my == y2 :
                print(dp[x][y] + 1)
                state = False
                break

            if dp[mx][my] == int(1e9) :
                dp[mx][my] = dp[x][y] + 1
                q.append([mx, my])

            k += 1
            mx += dx[i]
            my += dy[i]

        if not state :
            break

if state :
    print(-1)
```

<br>

#### Cribboard

``` Cribboard
N = int(input())
dp = [0] * 101
dp[1] = 1

for i in range(1, 101) :
    index = 2
    for j in range(index + i, 101) :
        if 101 > j + 1 :
            dp[j + 1] = max(dp[j + 1], dp[i] * index)
            index += 1

    index = 0
    for j in range(i, 101) :
        if 101 > index + j + 1 :
            dp[index + j + 1] = max(dp[index + j + 1], dp[i] + index + 1)
            index += 1
print(dp[N])
```

<br>

#### 1 Grade

``` 1 grade
import sys

N = int((sys.stdin.readline()).strip())
numbers = list(map(int, (sys.stdin.readline()).split()))

dp = [[0 for _ in range(0, 21)] for _ in range(100)]
dp[0][numbers[0]] = 1

for i in range(1, N - 1) :
    target = numbers[i]
    for j in range(21) :
        if 0 <= j + target <= 20 :
            dp[i][j + target] += dp[i - 1][j]
        if 0 <= j - target <= 20 :
            dp[i][j - target] += dp[i - 1][j]

print(dp[N - 2][numbers[N - 1]])
```

<br>

#### small locomotive

```small locomotive
import sys

N = int(sys.stdin.readline().strip())
train = list(map(int, (sys.stdin.readline().split())))
K = int(sys.stdin.readline().strip())

people_sum, sum_value = [], sum(train[0 : K])
people_sum.append(sum_value)
for i in range(1, N - K + 1) :
    sum_value -= train[i - 1]
    sum_value += train[i + K - 1]
    people_sum.append(sum_value)

dp = [[0 for _ in range(N)] for _ in range(3)]
for i in range(1, N - K + 2) :
    dp[0][i] = max(dp[0][i - 1], people_sum[i - 1])

for i in range(1, 3) :
    for j in range(1 + (i * K), N - K + 2) :
        dp[i][j] = max(dp[i][j - 1], dp[i - 1][j - K] + people_sum[j - 1])

print(max(dp[2]))
```

<br>

#### bracket

``` bracket
# catalan수는 n개의 노드를 이용해 이진트리를 만드는 경우의 수를 풀때도 사용됩니다.

import math, sys


def catalan(number) :
    standard = math.factorial(number)
    return math.factorial(number * 2) // (standard ** 2 * (number + 1))


T = int(sys.stdin.readline().strip())

for _ in range(T) :
    L = int(sys.stdin.readline().strip())
    if L % 2 == 0 :
        print(catalan(L // 2) % 1000000007)
    else :
        print(0)
```

<br>

#### Mutalisk

``` Mutalisk
import sys
from itertools import permutations
from collections import deque

N = int(sys.stdin.readline().strip())
scv_list = list(map(int, (sys.stdin.readline()).split()))

if N == 1 :
    value = scv_list[0]
    result = value // 9
    if value % 9 != 0 :
        result += 1
    print(result)
elif N == 2 :
    scv_list.append(0)
    q = deque()
    q.append(scv_list)

    while q :
        scv_a, scv_b, dist = q.popleft()

        if scv_a == 0 and scv_b == 0 :
            print(dist)
            break

        for a, b in permutations([9, 3], 2) :
            next_scv = []
            if a >= scv_a :
                next_scv.append(0)
            else :
                next_scv.append(scv_a - a)

            if b >= scv_b :
                next_scv.append(0)
            else :
                next_scv.append(scv_b - b)

            next_scv.append(dist + 1)
            q.append(next_scv)
else :
    scv_list.append(0)
    q = deque()
    q.append(scv_list)

    while q:
        scv_a, scv_b, scv_c, dist = q.popleft()

        if scv_a == 0 and scv_b == 0 and scv_c == 0:
            print(dist)
            break

        for a, b, c in permutations([9, 3, 1], 3):
            next_scv = []
            if a >= scv_a:
                next_scv.append(0)
            else:
                next_scv.append(scv_a - a)

            if b >= scv_b:
                next_scv.append(0)
            else:
                next_scv.append(scv_b - b)

            if c >= scv_c:
                next_scv.append(0)
            else:
                next_scv.append(scv_c - c)

            next_scv.append(dist + 1)

            if next_scv not in q :
                q.append(next_scv)
                
```

<br>

#### LCD Test

``` LCD Test
import sys

N = int(sys.stdin.readline().strip())
signal = list(sys.stdin.readline().strip())
standard, current, result = N // 5, 0, ""

while standard > current :
    if signal[current] == "#" :
        if signal[current + 1] == "#" and standard > current + 1 :
            if signal[current + 1 + (standard * 2)] == "#" :
                if signal[current + 2 + (standard * 1)] == "#" :
                    if signal[current + (standard * 3)] == "#" :
                        if signal[current + (standard * 1)] == "#" :
                            result += "8"
                            current += 4
                        else :
                            result += "2"
                            current += 4
                    else :
                        if signal[current + (standard * 1)] == "#" :
                            result += "9"
                            current += 4
                        else :
                            result += "3"
                            current += 4
                else :
                    if signal[current + (standard * 3)] == "#" :
                        result += "6"
                        current += 4
                    else :
                        result += "5"
                        current += 4
            else :
                if signal[current + 1 + (standard * 4)] == "#" :
                    result += "0"
                    current += 4
                else :
                    result += "7"
                    current += 4
        else :
            if signal[current + 1 + (standard * 2)] == "#" and standard > current + 1:
                result += "4"
                current += 4
            else :
                result += "1"
                current += 2
    else :
        current += 1

print(result)
```

<br>

#### World Cup

```World Cup
import copy, sys
from itertools import combinations


def dfs(copy_game, visited, standard):
    if standard == 6:
        total = 0
        for t in range(6):
            total += copy_game[t * 3 + 1] + copy_game[t * 3 + 2]
        if total == 0:
            global state
            state = True
            return "exit"
        else:
            return

    if copy_game[standard * 3] == 0:
        if copy_game[standard * 3 + 1] == 0:
            if dfs(copy_game, visited, standard + 1) == "exit":
                return "exit"
        else:
            for draw_comb in combinations(visited[standard], copy_game[standard * 3 + 1]):
                d_game, copy_draw, draw_state = copy.deepcopy(copy_game), copy.deepcopy(visited), True
                for d in draw_comb:
                    if d_game[d * 3 + 1] > 0 and d in copy_draw[standard]:
                        d_game[standard * 3 + 1] -= 1
                        d_game[d * 3 + 1] -= 1
                        copy_draw[standard].remove(d)
                        if standard in copy_draw[d] :
                            copy_draw[d].remove(standard)
                    else:
                        draw_state = False
                        break
                if draw_state:
                    if dfs(d_game, copy_draw, standard + 1) == "exit":
                        return "exit"
    else:
        for comb in combinations(visited[standard], copy_game[standard * 3]):
            c_game, copy_draw, win_loss_state = copy.deepcopy(copy_game), copy.deepcopy(visited), True
            for c in comb:
                if c_game[c * 3 + 2] > 0 and c in copy_draw[standard]:
                    c_game[c * 3 + 2] -= 1
                    copy_draw[standard].remove(c)
                    if standard in copy_draw[c] :
                        copy_draw[c].remove(standard)
                else:
                    win_loss_state = False
                    break

            if win_loss_state:
                draw_amount = copy_game[standard * 3 + 1]
                if draw_amount == 0:
                    if dfs(c_game, copy_draw, standard + 1) == "exit":
                        return "exit"
                elif len(copy_draw[standard]) >= draw_amount:
                    for draw_comb in combinations(copy_draw[standard], draw_amount):
                        d_c_game, d_c_draw, draw_state = copy.deepcopy(c_game), copy.deepcopy(copy_draw), True
                        for d in draw_comb :
                            if d_c_game[d * 3 + 1] > 0 and d in d_c_draw[standard]:
                                d_c_game[standard * 3 + 1] -= 1
                                d_c_game[d * 3 + 1] -= 1
                                d_c_draw[standard].remove(d)
                                if standard in d_c_draw[d] :
                                    d_c_draw[d].remove(standard)
                            else:
                                draw_state = False
                        if draw_state:
                            if dfs(d_c_game, d_c_draw, standard + 1) == "exit":
                                return "exit"


result = []
for _ in range(4):
    game = list(map(int, (sys.stdin.readline()).split()))
    state, sum_state, visited = False, True, [list(range(6)) for _ in range(6)]
    for i in range(6) :
        visited[i].remove(i)
    for s in range(6):
        if sum(game[(s * 3): (s * 3) + 3]) != 5:
            sum_state = False
            break

    if sum_state :
        dfs(game, visited, 0)

    if state:
        result.append(1)
    else:
        result.append(0)

for i in result:
    print(i, end=" ")
```

<br>

#### turtle

```turtle
import sys

T = int(sys.stdin.readline())
dx, dy = [0, 1, 0, -1], [1, 0, -1, 0]
for _ in range(T) :
    order = list(sys.stdin.readline().strip())
    history, current_point, location = [[0, 0]], [0, 0], 0

    for i in order :
        if i == "L" :
            location = (location - 1) % 4
        elif i == "R" :
            location = (location + 1) % 4
        elif i == "F" :
            current_point = [current_point[0] + dx[location], current_point[1] + dy[location]]
            history.append(current_point)
        elif i == "B" :
            current_point = [current_point[0] - dx[location], current_point[1] - dy[location]]
            history.append(current_point)

    max_0, min_0 = 0, int(1e9)
    max_1, min_1 = 0, int(1e9)
    for a, b in history :
        if a > max_0 :
            max_0 = a
        if min_0 > a :
            min_0 = a
        if b > max_1:
            max_1 = b
        if min_1 > b:
            min_1 = b
    print((max_0 - min_0) * (max_1 - min_1))
```

<br>

#### prefix

``` prefix
import sys


def check(a, b) :
    for w in range(len(a), len(b) + 1) :
        if b[:w] == a :
            return True
    return False


N = int(sys.stdin.readline().strip())
words = []
for _ in range(N) :
    words.append(sys.stdin.readline().strip())

words = sorted(words, key=lambda x : len(x))
total = 0
for i in range(N) :
    state = True
    for j in range(i + 1, N) :
        if check(words[i], words[j]) :
            state = False
            break
    if state :
        total += 1
print(total)
```

<br>

#### app

``` app
import sys

N, M = map(int, (sys.stdin.readline().split()))
A = [0] + list(map(int, (sys.stdin.readline().split())))
c = [0] + list(map(int, (sys.stdin.readline().split())))

dp = [[0 for _ in range(sum(c) + 1)] for _ in range(N + 1)]
result = int(1e9)

for i in range(1, N + 1) :
    for j in range(1, sum(c) + 1) :
        weight, value = c[i], A[i]

        if j < weight :
            dp[i][j] = dp[i - 1][j]
        else :
            dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight] + value)

        if dp[i][j] >= M :
            result = min(result, j)

print(result)
```

배운점
 - Knapsack인걸 눈치채지 못했다.
 - 어느 공간에 가치와 무게를 가지고 가치에 대한 최대무게는? 이라는 문제가 나올때 이형식을 따라가자

<br>

#### Puyo Puyo

``` puyo puyo
import sys
from collections import deque


def check() :
    global board
    visited = [[True for _ in range(6)] for _ in range(12)]
    trans_state = False

    for i in range(12) :
        for j in range(6) :
            if visited[i][j] and board[i][j] != ".":
                standard_color = board[i][j]
                q, visited[i][j], blocks = deque(), False, []
                q.append([i, j])

                while q :
                    x, y = q.popleft()
                    if [x, y] not in blocks :
                        blocks.append([x, y])

                    for k in range(4) :
                        mx, my = dx[k] + x, dy[k] + y
                        if 0 <= mx < 12 and 0 <= my < 6 and visited[mx][my] and standard_color == board[mx][my] :
                            visited[mx][my] = False
                            if [mx, my] not in q :
                                q.append([mx, my])

                if len(blocks) >= 4 :
                    trans_state = True
                    for x, y in blocks :
                        board[x][y] = "."
    return trans_state


def gravity() :
    global board

    for i in range(6) :
        keep = []
        for j in range(11, -1, -1) :
            if board[j][i] != "." :
                keep.append(board[j][i])
        keep_len = len(keep)
        for j in range(keep_len) :
            board[11 - j][i] = keep[j]
        for j in range(12 - keep_len) :
            board[j][i] = "."


board, total = [], 0
for _ in range(12) :
    board.append(list(sys.stdin.readline().strip()))

dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]

while check() :
    total += 1
    gravity()

print(total)
```

<br>

#### RGB distance

``` RGB distance
import copy, sys

N = int(sys.stdin.readline().strip())
color_cost, dp = [], [[int(1e9), int(1e9), int(1e9)] for _ in range(N)]
for _ in range(N) :
    color_cost.append(list(map(int, (sys.stdin.readline()).split())))

dp[0] = copy.deepcopy(color_cost[0])
for i in range(1, N) :
    for j in range(3) :
        if j == 0 :
            dp[i][j] = min(dp[i - 1][1] + color_cost[i][j], dp[i - 1][2] + color_cost[i][j])
        if j == 1 :
            dp[i][j] = min(dp[i - 1][0] + color_cost[i][j], dp[i - 1][2] + color_cost[i][j])
        if j == 2 :
            dp[i][j] = min(dp[i - 1][0] + color_cost[i][j], dp[i - 1][1] + color_cost[i][j])

print(min(dp[N - 1]))
```

<br>

#### Int Triangle

``` int triangle
import sys

n = int(sys.stdin.readline().strip())
tri, dp = [], [[0 for _ in range(i + 1)] for i in range(n)]
for _ in range(n) :
    tri.append(list(sys.stdin.readline().split()))

dp[0][0] = int(tri[0][0])
for i in range(1, n) :
    for j in range(i) :
        dp[i][j] = max(dp[i][j], dp[i - 1][j] + int(tri[i][j]))
        dp[i][j + 1] = max(dp[i][j + 1], dp[i - 1][j] + int(tri[i][j + 1]))

print(max(dp[n - 1]))
```

<br>

#### wine tasting

```wine tasting
import sys

n = int(sys.stdin.readline().strip())
ju, dp = [], [0] * n
for _ in range(n) :
    ju.append(int(sys.stdin.readline().strip()))

if n <= 2 :
    print(sum(ju))
elif n == 3 :
    print(max(ju[0] + ju[2], ju[1] + ju[2], ju[0] + ju[1]))
else :
    dp[0], dp[1] = ju[0], ju[0] + ju[1]
    dp[2] = max(ju[0] + ju[2], ju[1] + ju[2])
    dp[3] = max(ju[3] + ju[2] + ju[0], ju[3] + ju[1] + ju[0])

    for i in range(4, n) :
        dp[i] = ju[i] + max(dp[i - 2], ju[i - 1] + dp[i - 3], ju[i - 1] + dp[i - 4])

    print(max(dp))
```

<br>

#### 2 x n tiling 2

``` 2 x n tiling 2
import sys

n = int(sys.stdin.readline().strip())
dp = [0] * 1001
dp[1], dp[2] = 1, 3

for i in range(3, 1001) :
    dp[i] = (dp[i - 1] + dp[i - 2] + dp[i - 2]) % 10007

print(dp[n])
```

<br>

#### KMP

``` kmp
def solution(allString, pattern) :
    pattern_size = len(pattern)
    table = [0 for _ in range(pattern_size)]

    i = 0
    for j in range(1, pattern_size) :
        while i > 0 and pattern[i] != pattern[j] :
            i = table[i - 1]
        if pattern[i] == pattern[j] :
            i += 1
            table[j] = i

    result = []
    i = 0
    for j in range(len(allString)) :
        while i > 0 and pattern[i] != allString[j] :
            i = table[i - 1]
        if pattern[i] == allString[j] :
            i += 1
            if i == pattern_size :
                result.append(j - i + 1)
                i = table[i - 1]
                
    return result


allString = "xabxxbaxbaxbaxbaxabxbaxbabx"
pattern = "abx"
print(solution(allString, pattern))
```

<br>

#### string explosion

```string explosion
import sys

allString = sys.stdin.readline().strip()
pattern = sys.stdin.readline().strip()
pattern_size = len(pattern)
q, pattern_end = [], pattern[pattern_size - 1]

for i in allString :
    q.append(i)
    if i == pattern_end and "".join(q[-pattern_size:]) == pattern :
        del q[-pattern_size:]

if q :
    print("".join(q))
else :
    print("FRULA")
```

배운점
 - 문자열 문제는 KMP 알고리즘을 활용하는 것과 스택을 사용하는 것 2가지 경우로 나뉜다.
 - 배열의 [-상수 :]의 의미와 del q[상수:]의 활용도 알아두자

<br>

#### sticker

``` sticker
import sys

T = int(sys.stdin.readline().strip())
for _ in range(T) :
    n = int(sys.stdin.readline().strip())
    max_value = 0
    dp = [[0, 0] for _ in range(n)]
    first_s, second_s = list(map(int, (sys.stdin.readline()).split())), list(map(int, (sys.stdin.readline()).split()))

    if n == 1 :
        print(max(first_s[0], second_s[0]))
        continue

    dp[0] = [first_s[0], second_s[0]]
    dp[1] = [first_s[1] + second_s[0], first_s[0] + second_s[1]]
    for i in range(2, n) :
        dp[i][0] = max(dp[i][0], first_s[i] + dp[i - 1][1], first_s[i] + dp[i - 2][0], first_s[i] + dp[i - 2][1])
        dp[i][1] = max(dp[i][1], second_s[i] + dp[i - 1][0], second_s[i] + dp[i - 2][0], second_s[i] + dp[i - 2][1])

    for values in dp :
        value = max(values)
        if value > max_value :
            max_value = value

    print(max_value)
```

<br>

#### find

``` find
import sys


def kmp(allString, pattern) :
    pattern_size = len(pattern)
    table = [0 for _ in range(pattern_size)]
    i = 0
    for j in range(1, pattern_size) :
        while i > 0 and pattern[i] != pattern[j] :
            i = table[i - 1]
        if pattern[i] == pattern[j] :
            i += 1
            table[j] = i

    result = []
    i = 0
    for j in range(len(allString)) :
        while i > 0 and pattern[i] != allString[j] :
            i = table[i - 1]
        if pattern[i] == allString[j] :
            i += 1
            if i == pattern_size :
                result.append(j - i + 1)
                i = table[i - 1]

    return result


allString = input()
pattern = input()
result = kmp(allString, pattern)
print(len(result))
for i in result :
    print(i + 1, end=" ")
```

<br>

#### robotic vacuum

``` robotic vacuum
import sys

N, M = map(int, (sys.stdin.readline()).split())
current_x, current_y, direction = map(int, (sys.stdin.readline()).split())

board = []
dx, dy = [-1, 0, 1, 0], [0, 1, 0, -1]
for _ in range(N) :
    board.append(list(map(int, (sys.stdin.readline()).split())))

result, state_index, dist = int(1e9), 0, 1
board[current_x][current_y] = 2

while True :
    if state_index >= 4 :
        mx, my = current_x - dx[direction], current_y - dy[direction]
        if board[mx][my] == 1 :
            result = dist
            break
        else :
            current_x, current_y, state_index = mx, my, 0
            continue

    direction = (direction - 1 + 4) % 4
    mx, my = current_x + dx[direction], current_y + dy[direction]
    if 0 <= mx < N and 0 <= my < M and board[mx][my] == 0:
        board[mx][my] = 2
        current_x, current_y, state_index, dist = mx, my, 0, dist + 1
        continue
    else:
        state_index += 1

print(result)
```

<br>

#### number of climbs

```number of climbs
import sys

N = int(sys.stdin.readline().strip())

dp = [[0 for _ in range(10)] for _ in range(1001)]
dp[1] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

for i in range(2, 1001) :
    for j in range(10) :
        for k in range(j, 10) :
            dp[i][k] += dp[i - 1][j]

print(sum(dp[N]) % 10007)
```

<br>

#### party

``` party
import sys, heapq

N, M, X = map(int, (sys.stdin.readline()).split())

edges, max_distance = [[] for _ in range(N + 1)], 0
for _ in range(M) :
    s, e, t = map(int, (sys.stdin.readline()).split())
    edges[s].append([e, t])


def dijkstra(start) :
    distances = [int(1e9) for _ in range(N + 1)]
    distances[start] = 0

    q = []
    heapq.heappush(q, [0, start])
    while q :
        dist, node = heapq.heappop(q)

        if dist > distances[node] :
            continue

        for i in edges[node] :
            cost = dist + i[1]
            if distances[i[0]] > cost :
                heapq.heappush(q, [cost, i[0]])
                distances[i[0]] = cost

    return distances


edge_distances = [[int(1e9) for _ in range(N + 1)]]
for i in range(1, N + 1) :
    edge_distances.append(dijkstra(i))

for i in range(1, N + 1) :
    total_distance = edge_distances[i][X] + edge_distances[X][i]
    if int(1e9) > total_distance :
        max_distance = max(max_distance, total_distance)
        
print(max_distance)
```

<br>

#### network

``` network
import sys, heapq


def find_parent(parent, x) :
    if parent[x] != x :
        return find_parent(parent, parent[x])
    return x


def union_parent(parent, x, y) :
    x = find_parent(parent, x)
    y = find_parent(parent, y)

    if x > y :
        parent[x] = y
    else :
        parent[y] = x


N = int(sys.stdin.readline().strip())
M = int(sys.stdin.readline().strip())

q, parent = [], [i for i in range(N + 1)]

for _ in range(M) :
    a, b, c = map(int, (sys.stdin.readline()).split())
    if a != b :
        heapq.heappush(q, [c, a, b])

total = 0
while q :
    cost, a, b = heapq.heappop(q)
    if find_parent(parent, a) != find_parent(parent, b) :
        union_parent(parent, a, b)
        total += cost

print(total)
```

<br>

#### top

``` top
import sys, heapq

N = int(sys.stdin.readline().strip())
towers = list(map(int, (sys.stdin.readline()).split()))
result = []

q = []
for i in range(N - 1, -1, -1) :
    while q and towers[i] > q[0][0] :
        result.append([i + 1, heapq.heappop(q)[1]])
    heapq.heappush(q, [towers[i], i + 1])

for i in q :
    result.append([0, i[1]])

result = sorted(result, key=lambda x : x[1])
for a, b in result :
    print(a, end=" ")
```

<br>

#### greedy pandas

``` greedy pandas
import sys
sys.setrecursionlimit(int(1e6))

n = int(sys.stdin.readline().strip())
board, max_distance = [], 0
dx, dy = [0, 0, 1, -1], [1, -1, 0, 0]
dp = [[1 for _ in range(n)] for _ in range(n)]

for _ in range(n) :
    board.append(list(map(int, (sys.stdin.readline()).split())))


def dfs(x, y):
    if dp[x][y] != 1 :
        return dp[x][y]

    for k in range(4) :
        mx, my = x + dx[k], y + dy[k]
        if 0 <= mx < n and 0 <= my < n and board[mx][my] > board[x][y] :
            dp[x][y] = max(dp[x][y], dfs(mx, my) + 1)

    return dp[x][y]


for i in range(n) :
    for j in range(n) :
        dp[i][j] = max(dp[i][j], dfs(i, j))

for i in range(n) :
    max_distance = max(max_distance, max(dp[i]))

print(max_distance)
```

<br>

#### multiplication

``` multiplication
import sys

A, B, C = map(int, (sys.stdin.readline()).split())


def mu(a, b) :
    if b == 1 :
        return a % C
    elif b == 0 :
        return 1 % C

    if b % 2 == 0 :
        return (mu(a, b // 2) ** 2) % C
    else :
        return (mu(a, b // 2) ** 2 * a) % C


print(mu(A, B))
```

<br>

#### sum of squares

``` sum of squares
import sys

N = int(sys.stdin.readline().strip())
dp = [i for i in range(N + 1)]
square_number = [i ** 2 for i in range(1, int(N ** 0.5) + 1)]

for i in range(1, N + 1) :
    for j in square_number :
        if j > i:
            break
        if dp[i] > dp[i - j] + 1 :
            dp[i] = dp[i - j] + 1

print(dp[N])
```

<br>

#### kaying calendar

``` kaying calendar
import sys

T = int(sys.stdin.readline().strip())


def gcd(n, m) :
    mod = m % n
    if mod != 0 :
        m, n = n, mod
        return gcd(n, m)
    else:
        return n


def lcm(n, m) :
    return int(n*m / gcd(n, m))


for _ in range(T) :
    M, N, x, y = map(int, (sys.stdin.readline()).split())
    lcm_mn = lcm(M, N)
    day, total, state = 0, 0, True
    if x > y :
        sub = x - y
        while sub != day :
            day = (day + N) % M
            total += N
            if total > lcm_mn :
                state = False
                break

        if state :
            print(total + y)
        else :
            print(-1)
    else :
        sub = y - x
        while sub != day :
            day = (day + M) % N
            total += M
            if total > lcm_mn :
                state = False
                break

        if state:
            print(total + x)
        else:
            print(-1)
```

<br>

#### matrix squared

``` matrix squared
import sys


def mutiple(A, B) :
    A_len = len(A)
    temp = [[0 for _ in range(A_len)] for _ in range(A_len)]

    for i in range(A_len) :
        for j in range(A_len) :
            for k in range(A_len) :
                temp[i][j] += A[i][k] * B[k][j]

    for i in range(A_len) :
        for j in range(A_len) :
            temp[i][j] %= 1000

    return temp


def mod_squared(A, B) :
    if B == 1 :
        for i in range(len(A)):
            for j in range(len(A)):
                A[i][j] %= 1000
        return A

    elif B == 2 :
        return mutiple(A, A)

    if B % 2 == 0 :
        result = mod_squared(A, B // 2)
        return mutiple(result, result)
    else :
        result = mod_squared(A, B // 2)
        return mutiple(mutiple(result, result), A) # B행렬 곱하기기


N, B = map(int, (sys.stdin.readline()).split())
A = []

for _ in range(N) :
    A.append(list(map(int, (sys.stdin.readline()).split())))

result = mod_squared(A, B)

for i in range(len(result)):
    for j in range(len(result)):
        print(result[i][j], end=" ")
    print()
```

<br>

#### decimal path

``` decimal path
import copy, sys
from collections import deque

state = [True for _ in range(10000)]

for i in range(2, int(10000 ** 0.5) + 1):
    if state[i]:
        for j in range(i + i, 10000, i):
            state[j] = False

primeNumber = {str(i) : True for i in range(1000, 10000) if state[i]}
prime = list(primeNumber.keys())

T = int(sys.stdin.readline())
for _ in range(T) :
    A, B = map(int, (sys.stdin.readline()).split())

    visited = copy.deepcopy(primeNumber)
    visited[str(A)], max_distance = False, 0
    q, A, B = deque(), str(A), str(B)
    q.append([str(A), 1])
    state = False

    while q :
        current, dist = q.popleft()

        for k in range(4) :
            for n in range(10) :
                trans = current[:k] + str(n) + current[k + 1:]
                if trans == B :
                    max_distance, state = dist, True
                    break

                if trans in prime and visited[trans] :
                    visited[trans] = False
                    q.append([trans, dist + 1])

            if state :
                break
        if state :
            break

    if not state:
        print("Impossible")
    elif A == B :
        print(0)
    else :
        print(max_distance)
```

<br>

#### Find the sum of intervals 5

```Find the sum of intervals 5
import sys

N, M = map(int, (sys.stdin.readline()).split())

board = [[0 for _ in range(N + 1)]]
for _ in range(N):
    board.append([0] + list(map(int, (sys.stdin.readline()).split())))

for i in range(1, N + 1):
    for j in range(1, N):
        board[i][j + 1] += board[i][j]

for j in range(1, N + 1):
    for i in range(1, N):
        board[i + 1][j] += board[i][j]

for _ in range(M):
    y1, x1, y2, x2 = map(int, (sys.stdin.readline()).split())
    print(board[y2][x2] - board[y1 - 1][x2] - board[y2][x1 - 1] + board[y1 - 1][x1 - 1])
```

<br>

#### fish king

``` fish king
import copy, sys

R, C, M = map(int, (sys.stdin.readline()).split())
board = [[[] for _ in range(C + 1)] for _ in range(R + 1)]
total = 0

dx, dy = [-1, 1, 0, 0], [0, 0, 1, -1]

for _ in range(M) :
    r, c, s, d, z = map(int, (sys.stdin.readline()).split())
    board[r][c].append([s, d, z])

for i in range(1, C + 1) :
    for j in range(1, R + 1) :
        if board[j][i]:
            total += board[j][i].pop()[2]
            break

    temp_board = [[[] for _ in range(C + 1)] for _ in range(R + 1)]
    for a in range(1, R + 1) :
        for b in range(1, C + 1) :
            if board[a][b] :
                s, d, z = board[a][b].pop()
                mx, my = a + dx[d - 1] * s, b + dy[d - 1] * s
                if d > 2 :
                    if my < 0:
                        my = -(-my % ((C - 1) * 2))
                    else:
                        my %= (C - 1) * 2
                    state = 0
                    if my <= 0 :
                        state = 1
                        my = -(my - 2)
                    if my > C:
                        state += (my - 2) // (C - 1)
                        if state % 2 == 1 :
                            my = C - ((my - 2) % (C - 1) + 1)
                        else :
                            if d == 3 :
                                my = (my - 2) % (C - 1) + 2
                            else :
                                my = C - ((my - 2) % (C - 1) + 1)

                    if state % 2 == 0 :
                        temp_board[mx][my].append([s, d, z])
                    else :
                        if d == 3 :
                            d = 4
                        else :
                            d = 3
                        temp_board[mx][my].append([s, d, z])
                else :
                    if mx < 0:
                        mx = -(-mx % ((R - 1) * 2))
                    else:
                        mx %= (R - 1) * 2
                    state = 0
                    if mx <= 0 :
                        state = 1
                        mx = -(mx - 2)
                    if mx > R:
                        state += (mx - 2) // (R - 1)
                        if state % 2 == 1 :
                            mx = R - ((mx - 2) % (R - 1) + 1)
                        else :
                            if d == 2 :
                                mx = (mx - 2) % (R - 1) + 2
                            else :
                                mx = R - ((mx - 2) % (R - 1) + 1)

                    if state % 2 == 0:
                        temp_board[mx][my].append([s, d, z])
                    else :
                        if d == 1 :
                            d = 2
                        else :
                            d = 1
                        temp_board[mx][my].append([s, d, z])
                if len(temp_board[mx][my]) >= 2 :
                    temp_board[mx][my] = sorted(temp_board[mx][my], key=lambda x : -x[2])
                    temp_board[mx][my] = [temp_board[mx][my][0]]

    board = copy.deepcopy(temp_board)
print(total)
```

<br>

#### runway

```runway
import sys

N, L = map(int, (sys.stdin.readline()).split())
board, count = [], 0

for _ in range(N) :
    board.append(list(map(int, (sys.stdin.readline()).split())))


def check(type, index):
    # 가로줄 탐색
    if type == 0:
        current_height, same_height, state, j = board[index][0], 1, True, 1
        while N > j and state:
            if current_height == board[index][j]:
                same_height += 1
            elif current_height + 1 == board[index][j] and same_height >= L :
                current_height, same_height = board[index][j], 1
            elif current_height - 1 == board[index][j] and N - j >= L:
                for k in range(j + 1, j + L) :
                    if current_height - 1 != board[index][k] :
                        state = False
                        break
                if state :
                    current_height, same_height = board[index][j], 0
                    j += (L - 1)
            else :
                state = False
            j += 1

        if not state :
            current_height, same_height, state, j = board[index][N - 1], 1, True, N - 2
            while j >= 0 and state:
                if current_height == board[index][j]:
                    same_height += 1
                elif current_height + 1 == board[index][j] and same_height >= L:
                    current_height, same_height = board[index][j], 1
                elif current_height - 1 == board[index][j] and j >= L - 1:
                    for k in range(j - 1, j - L - 1, -1):
                        if current_height - 1 != board[index][k]:
                            return False
                    if state:
                        current_height, same_height = board[index][j], 0
                        j -= (L - 1)
                else:
                    return False
                j -= 1
        return True
    else :
        current_height, same_height, state, j = board[0][index], 1, True, 1
        while N > j and state:
            if current_height == board[j][index]:
                same_height += 1
            elif current_height + 1 == board[j][index] and same_height >= L:
                current_height, same_height = board[j][index], 1
            elif current_height - 1 == board[j][index] and N - j >= L:
                for k in range(j + 1, j + L):
                    if current_height - 1 != board[k][index]:
                        state = False
                        break
                if state:
                    current_height, same_height = board[j][index], 0
                    j += (L - 1)
            else:
                state = False
            j += 1

        if not state:
            current_height, same_height, state, j = board[N - 1][index], 1, True, N - 2
            while j >= 0 and state:
                if current_height == board[j][index]:
                    same_height += 1
                elif current_height + 1 == board[j][index] and same_height >= L:
                    current_height, same_height = board[j][index], 1
                elif current_height - 1 == board[j][index] and j >= L - 1:
                    for k in range(j - 1, j - L - 1, -1):
                        if current_height - 1 != board[k][index]:
                            return False
                    if state:
                        current_height, same_height = board[j][index], 0
                        j -= (L - 1)
                else:
                    return False
                j -= 1
        return True


for i in range(N) :
    if check(0, i) :
        count += 1
    if check(1, i) :
        count += 1

print(count)
```

<br>

#### tree finance

``` tree finance
import copy
import sys

N, M, K = map(int, (sys.stdin.readline()).split())
A, board = [], [[[[], 5] for _ in range(N)] for _ in range(N)]

for _ in range(N) :
    A.append(list(map(int, (sys.stdin.readline()).split())))

for _ in range(M) :
    x, y, z = map(int, (sys.stdin.readline()).split())
    board[x - 1][y - 1][0].append(z)

dx, dy = [1, -1, 0, 0, -1, 1, 1, -1], [0, 0, 1, -1, -1, 1, -1, 1]

for _ in range(K) :
    # 봄, 여름
    for i in range(N) :
        for j in range(N) :
            if board[i][j][0] :
                target, food = sorted(board[i][j][0]), board[i][j][1]

                k, add_food, target_len, state = 0, 0, len(target), True
                for k in range(target_len) :
                    food -= target[k]
                    if food < 0 :
                        food += target[k]
                        state = False
                        break
                    target[k] += 1
                if not state :
                    for d in range(k, target_len) :
                        add_food += target[d] // 2
                    k -= 1
                board[i][j][0] = copy.deepcopy(target[:k + 1])
                board[i][j][1] = food + add_food

    # 가을, 겨울
    for i in range(N) :
        for j in range(N) :
            if board[i][j][0] :
                target = board[i][j][0]
                target_len = len(target)
                for k in range(target_len) :
                    if target[k] % 5 == 0 :
                        for s in range(8) :
                            mx, my = i + dx[s], j + dy[s]
                            if 0 <= mx < N and 0 <= my < N :
                                board[mx][my][0].append(1)
            board[i][j][1] += A[i][j]

result = 0
for i in range(N) :
    for j in range(N) :
        if board[i][j] :
            result += len(board[i][j][0])

print(result)
```

<br>

#### cheeze

``` cheeze
import sys
from collections import deque

N, M = map(int, (sys.stdin.readline()).split())
board, count, state = [], 0, False
for _ in range(N) :
    board.append(list(map(int, (sys.stdin.readline()).split())))

dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]

for k in range(N):
    if sum(board[k]) != 0:
        state = True

if state :
    while True :
        q, state = deque(), True
        q.append([0, 0])
        visited, temp_board = [[True for _ in range(M)] for _ in range(N)], [[0 for _ in range(M)] for _ in range(N)]
        visited[0][0] = False
        while q:
            x, y = q.popleft()

            for i in range(4):
                mx, my = x + dx[i], y + dy[i]
                if 0 <= mx < N and 0 <= my < M and visited[mx][my] and board[mx][my] == 0:
                    visited[mx][my] = False
                    q.append([mx, my])
                    for j in range(4):
                        tx, ty = mx + dx[j], my + dy[j]
                        if 0 <= tx < N and 0 <= ty < M and board[tx][ty] == 1:
                            temp_board[tx][ty] += 1

        for i in range(N):
            for j in range(M):
                if temp_board[i][j] >= 2:
                    board[i][j] = 0
                elif board[i][j] != 0:
                    state = False

        count += 1
        if state :
            break

print(count)
```

<br>

#### Tree

``` Tree
class Node :
    def __init__(self, data, left_node, right_node):
        self.data = data
        self.left_node = left_node
        self.right_node = right_node


def pre_order(node) :
    print(node.data, end=" ")
    if node.left_node :
        pre_order(tree[node.left_node])
    if node.right_node :
        pre_order(tree[node.right_node])


def in_order(node) :
    if node.left_node :
        in_order(tree[node.left_node])
    print(node.data, end=" ")
    if node.right_node :
        in_order(tree[node.right_node])


def post_order(node) :
    if node.left_node :
        post_order(tree[node.left_node])
    if node.right_node :
        post_order(tree[node.right_node])
    print(node.data, end=" ")


n = int(input())
tree = {}
for i in range(n) :
    data, left_node, right_node = input().split()
    if left_node == "None" :
        left_node = None
    if right_node == "None":
        right_node = None

    tree[data] = Node(data, left_node, right_node)

pre_order(tree['A'])
print()
in_order(tree['A'])
print()
post_order(tree['A'])

# 7
# A B C
# B D E
# C F G
# D None None
# E None None
# F None None
# G None None
```

<br>

#### key

``` key
import sys
from collections import deque

T = int(sys.stdin.readline().strip())
dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]

for _ in range(T) :
    h, w = map(int, (sys.stdin.readline()).split())

    board, count, keys = [], 0, []
    for _ in range(h) :
        board.append(list(sys.stdin.readline().strip()))

    keys_str = sys.stdin.readline().strip()
    if keys_str == "0" :
        keys = []
    else :
        for i in list(keys_str) :
            keys.append(ord(i))

    q, doors = deque(), deque()
    visited = [[True for _ in range(w)] for _ in range(h)]
    for i in range(h) :
        if i == 0 or i == h - 1:
            for j in range(w) :
                if board[i][j] != "*" :
                    if 65 <= ord(board[i][j]) <= 90:
                        if ord(board[i][j]) + 32 not in keys :
                            doors.append([i, j, ord(board[i][j]) + 32])
                            continue
                    elif 97 <= ord(board[i][j]) <= 122 :
                        target = ord(board[i][j])
                        keys.append(target)
                        for d in range(len(doors) - 1, -1, -1) :
                            if doors[d][2] == target :
                                if [doors[d][0], doors[d][1]] not in q :
                                    q.append([doors[d][0], doors[d][1]])
                                    visited[doors[d][0]][doors[d][1]] = False
                                    doors.remove(doors[d])
                    elif board[i][j] == "$" :
                        count += 1
                    q.append([i, j])
                    visited[i][j] = False

        else :
            w_list = [0, w - 1]
            for j in w_list :
                if board[i][j] != "*":
                    if 65 <= ord(board[i][j]) <= 90:
                        if ord(board[i][j]) + 32 not in keys:
                            doors.append([i, j, ord(board[i][j]) + 32])
                            continue
                    elif 97 <= ord(board[i][j]) <= 122:
                        target = ord(board[i][j])
                        keys.append(target)
                        for d in range(len(doors) - 1, -1, -1):
                            if doors[d][2] == target:
                                if [doors[d][0], doors[d][1]] not in q:
                                    q.append([doors[d][0], doors[d][1]])
                                    visited[doors[d][0]][doors[d][1]] = False
                                    doors.remove(doors[d])
                    elif board[i][j] == "$":
                        count += 1
                    q.append([i, j])
                    visited[i][j] = False

    while q:
        x, y = q.popleft()

        for k in range(4):
            mx, my = x + dx[k], y + dy[k]
            if 0 <= mx < h and 0 <= my < w and board[mx][my] != "*" and visited[mx][my]:
                if 65 <= ord(board[mx][my]) <= 90:
                    if ord(board[mx][my]) + 32 not in keys:
                        doors.append([mx, my, ord(board[mx][my]) + 32])
                        continue
                elif 97 <= ord(board[mx][my]) <= 122:
                    target = ord(board[mx][my])
                    keys.append(target)
                    for d in range(len(doors) - 1, -1, -1):
                        if doors[d][2] == target:
                            if [doors[d][0], doors[d][1]] not in q:
                                q.append([doors[d][0], doors[d][1]])
                                visited[doors[d][0]][doors[d][1]] = False
                                doors.remove(doors[d])
                elif board[mx][my] == "$":
                    count += 1
                q.append([mx, my])
                visited[mx][my] = False

    print(count)
```

<br>

#### number list

``` number list
import sys


class Node(object) :
    def __init__(self, key, data=None):
        self.key = key
        self.data = data
        self.childrens = {}


class Trie :
    def __init__(self):
        self.head = Node(None)

    def insert(self, string):
        current_node = self.head

        for char in string :
            if char not in current_node.childrens :
                current_node.childrens[char] = Node(char)
            current_node = current_node.childrens[char]
            if current_node.data :
                return False
        current_node.data = string
        return True


t = int(sys.stdin.readline().strip())

for _ in range(t) :
    n = int(sys.stdin.readline().strip())
    trie, numbers, state = Trie(), [], True

    for _ in range(n) :
        numbers.append(sys.stdin.readline().strip())

    numbers = sorted(numbers, key=lambda x : (len(x), x))

    for number in numbers :
        if not trie.insert(number) :
            state = False
            break

    if state :
        print("YES")
    else :
        print("NO")
```

<br>

#### start taxi

``` start taxi
import sys
from collections import deque

N, M, f = map(int, (sys.stdin.readline()).split())
board, count, info, state = [], 0, [[[] for _ in range(N)] for _ in range(N)], True

for _ in range(N) :
    board.append(list(map(int, (sys.stdin.readline()).split())))

x, y = map(int, (sys.stdin.readline()).split())
x, y = x - 1, y - 1

dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]

for _ in range(M) :
    sx, sy, ex, ey = map(int, (sys.stdin.readline()).split())
    info[sx - 1][sy - 1].append([ex - 1, ey - 1])

while count != M :
    q = deque()
    q.append([x, y, 0])
    visited = [[True for _ in range(N)] for _ in range(N)]
    result = []

    while q :
        x, y, dist = q.popleft()

        if info[x][y] :
            if not result :
                result = [dist, x, y]
            elif result[0] == dist :
                if result[1] > x :
                    result = [dist, x, y]
                elif result[1] == x :
                    if result[2] > y :
                        result = [dist, x, y]
            else :
                break

        for i in range(4) :
            mx, my = x + dx[i], y + dy[i]
            if 0 <= mx < N and 0 <= my < N and visited[mx][my] and board[mx][my] != 1:
                visited[mx][my] = False
                if [mx, my] not in q :
                    q.append([mx, my, dist + 1])

    if result and f >= result[0] :
        f -= result[0]
        q = deque()
        q.append([result[1], result[2], 0])
        visited = [[True for _ in range(N)] for _ in range(N)]
        tx, ty = info[result[1]][result[2]][0]
        min_dist = int(1e9)

        while q:
            x, y, dist = q.popleft()

            if x == tx and y == ty :
                min_dist = dist
                break

            for i in range(4):
                mx, my = x + dx[i], y + dy[i]
                if 0 <= mx < N and 0 <= my < N and visited[mx][my] and board[mx][my] != 1:
                    visited[mx][my] = False
                    if [mx, my] not in q:
                        q.append([mx, my, dist + 1])

        if f >= min_dist :
            f += min_dist
            count += 1
            info[result[1]][result[2]] = []
        else :
            state = False
            break
    else :
        state = False
        break

if state :
    print(f)
else :
    print(-1)
```

<br>

#### start and link

```start and link
import sys
from itertools import combinations

N = int(sys.stdin.readline().strip())

board, N_list = [], [i for i in range(N)]
min_value = int(1e11)
for _ in range(N) :
    board.append(list(map(int, sys.stdin.readline().split())))

for comb in combinations(N_list, N // 2) :
    sum_value_1 = 0
    for c in comb :
        for cc in comb :
            sum_value_1 += board[c][cc]

    comb_2 = []
    for i in N_list :
        if i not in comb :
            comb_2.append(i)

    sum_value_2 = 0
    for c in comb_2:
        for cc in comb_2:
            sum_value_2 += board[c][cc]

    min_value = min(min_value, abs(sum_value_2 - sum_value_1))

print(min_value)
```

<br>

#### number of connected elements

```number of connected elements
import sys


def find_parent(parent, x) :
    if parent[x] != x :
        return find_parent(parent, parent[x])
    return x


def union_parent(parent, a, b) :
    a = find_parent(parent, a)
    b = find_parent(parent, b)

    if a > b :
        parent[a] = b
    else :
        parent[b] = a


N, M = map(int, (sys.stdin.readline()).split())

parent = [0] + [(i + 1) for i in range(N)]
for _ in range(M) :
    u, v = map(int, (sys.stdin.readline()).split())
    union_parent(parent, u, v)

dic = {}
for i in range(1, N + 1) :
    target = find_parent(parent, i)
    if target not in dic.keys() :
        dic[target] = 1

print(len(dic.keys()))
```

<br>

#### Find parent of tree

``` Find parent of tree
import sys
sys.setrecursionlimit(int(1e9))


class Tree :
    def __init__(self, data, parent, current, answer):
        answer[current] = parent

        for j in data[current] :
            if j != parent :
                Tree(data, current, j, answer)


N = int(sys.stdin.readline().strip())
edges = [[] for _ in range(N + 1)]
for _ in range(N - 1):
    a, b = map(int, (sys.stdin.readline()).split())
    edges[a].append(b)
    edges[b].append(a)

answer = [0 for _ in range(N + 1)]
tree = Tree(edges, None, 1, answer)
for i in range(2, N + 1) :
    print(answer[i])
```

<br>

#### tree order

``` tree order
import sys


class Tree :
    def __init__(self, datas, current):
        self.root = current
        if datas[current][0] == '.' :
            self.left_node = None
        else :
            self.left_node = Tree(datas, datas[current][0])

        if datas[current][1] == '.':
            self.right_node = None
        else:
            self.right_node = Tree(datas, datas[current][1])


N = int(sys.stdin.readline().strip())

edges_dict = {}
for _ in range(N):
    c, l, r = map(str, (sys.stdin.readline()).split())
    edges_dict[c] = [l, r]

tree = Tree(edges_dict, 'A')
pre_list, in_list, post_list = [], [], []


def preorder(tree, pre_list) :
    pre_list.append(tree.root)
    if tree.left_node :
        preorder(tree.left_node, pre_list)
    if tree.right_node :
        preorder(tree.right_node, pre_list)


def inorder(tree, in_list) :
    if tree.left_node :
        inorder(tree.left_node, in_list)
    in_list.append(tree.root)
    if tree.right_node :
        inorder(tree.right_node, in_list)


def postorder(tree, post_list) :
    if tree.left_node :
        postorder(tree.left_node, post_list)
    if tree.right_node :
        postorder(tree.right_node, post_list)
    post_list.append(tree.root)


preorder(tree, pre_list)
inorder(tree, in_list)
postorder(tree, post_list)

for i in range(N) :
    print(pre_list[i], end="")
print()
for i in range(N) :
    print(in_list[i], end="")
print()
for i in range(N) :
    print(post_list[i], end="")
print()
```

<br>

#### tree diameter

``` tree diameter
import sys
sys.setrecursionlimit(int(1e8))


class Tree :
    def __init__(self, datas, current):
        self.root = current
        self.child = []
        self.child_dist = []
        if current in datas.keys() :
            childrens = datas[current]
            for i in range(len(childrens)) :
                self.child.append(Tree(datas, childrens[i][0]))
                self.child_dist.append(childrens[i][1])

    def cal_dist(self, dist):
        max_dist = 0
        dist_lists = []
        for i in range(len(self.child)) :
            x, y = self.child[i].cal_dist(self.child_dist[i])
            dist_lists.append(x)
            max_dist = max(max_dist, y)

        dist_lists = sorted(dist_lists, reverse=True)
        one, two = 0, 0
        if len(dist_lists) == 1 :
            one = dist_lists[0]
        elif len(dist_lists) >= 2 :
            one, two = dist_lists[0], dist_lists[1]

        return dist + max(one, two), max(one + two, max_dist)


n = int(sys.stdin.readline().strip())
edges = {}
for _ in range(n - 1) :
    a, b, c = map(int, (sys.stdin.readline()).split())
    if a not in edges.keys() :
        edges[a] = []
    edges[a].append([b, c])

tree = Tree(edges, 1)
g, result = tree.cal_dist(0)
print(result)
```

<br>

#### pikachu

``` pikachu
import sys, re

r = re.compile('(pi|ka|chu)+')
answer = r.fullmatch(sys.stdin.readline().strip())

print("YES" if answer != None else "NO")
```

<br>

#### machine code

``` machine code
import sys, re

data = sys.stdin.readline().strip()
answer = 0
results = re.split('[A-Z]', data)
for i in range(1, len(results) - 1) :
    answer += (4 - ((len(results[i]) + 1) % 4)) % 4
print(answer)
```

br>

#### pronounce password

``` pronounce password
import sys, re

while True :
    data = sys.stdin.readline().strip()
    if data == "end" :
        break
    if not re.search('a|e|i|o|u', data) :
        print("<" + data + "> is not acceptable.")
        continue
    if re.search('([a|e|i|o|u]{3})|([^a|e|i|o|u]{3})', data) :
        print("<" + data + "> is not acceptable.")
        continue
    if re.search(r'([a-df-np-z])\1', data) :
        print("<" + data + "> is not acceptable.")
        continue
    print("<" + data + "> is acceptable.")
```

<br>

#### LIS 2

``` LIS 2
import sys, bisect

N = int(sys.stdin.readline().strip())
A = list(map(int, (sys.stdin.readline()).split()))

elements = [A[0]]
for i in range(1, N) :
    if A[i] > elements[-1] :
        elements.append(A[i])
    else :
        idx = bisect.bisect_left(elements, A[i])
        elements[idx] = A[i]
print(len(elements))
```

<br>

#### LIS 5

``` LIS 5
import sys, bisect

N = int(sys.stdin.readline().strip())
A = list(map(int, (sys.stdin.readline()).split()))
elements = [A[0]]
dp = [1 for _ in range(N)]

for i in range(1, N) :
    if A[i] > elements[-1] :
        elements.append(A[i])
        dp[i] = len(elements)
    else :
        idx = bisect.bisect_left(elements, A[i])
        elements[idx] = A[i]
        dp[i] = idx + 1

elements_len = len(elements)
print(elements_len)
ans = []

for i in range(N - 1, -1, -1) :
    if dp[i] == elements_len :
        ans.append(A[i])
        elements_len -= 1

    if elements_len < 1 :
        break

print(*ans[::-1])
```

<br>

#### longest bitonic subsequence

``` longest bitonic subsequence
import sys

N = int(sys.stdin.readline().strip())
A = list(map(int, (sys.stdin.readline()).split()))

x = [1 for _ in range(N)]
y = [1 for _ in range(N)]
for i in range(N) :
    for j in range(i) :
        if A[i] > A[j] :
            x[i] = max(x[j] + 1, x[i])
        if A[N - i - 1] > A[N - j - 1]:
            y[N - i - 1] = max(y[N - j - 1] + 1, y[N - i - 1])

result = 0
for i in range(N) :
    result = max(result, x[i] + y[i] - 1)
print(result)
```

<br>

#### Find the Interval Sum

``` Find the Interval Sum
import sys


def init(node, start, end) :
    if start == end :
        tree[node] = l[start]
    else :
        mid = (start + end) // 2
        tree[node] = init(node * 2, start, mid) + init(node * 2 + 1, mid + 1, end)
    return tree[node]


def subSum(node, start, end, left, right) :
    if start > right or end < left :
        return 0

    if left <= start and end <= right :
        return tree[node]

    mid = (start + end) // 2
    return subSum(node * 2, start, mid, left, right) + subSum(node * 2 + 1, mid + 1, end, left, right)


def update(node, start, end, index, diff) :
    if start > index or end < index :
        return

    tree[node] += diff

    if start != end :
        mid = (start + end) // 2
        update(node * 2, start, mid, index, diff)
        update(node * 2 + 1, mid + 1, end, index, diff)


N, M, K = map(int, (sys.stdin.readline()).split())
l = []
for _ in range(N) :
    l.append(int(sys.stdin.readline().strip()))

tree = [0] * (N * 5)
init(1, 0, N - 1)

for _ in range(M + K) :
    a, b, c = map(int, (sys.stdin.readline()).split())
    if a == 1 :
        diff = c - l[b - 1]
        l[b - 1] = c
        update(1, 0, N - 1, b - 1, diff)
    else :
        print(subSum(1, 0, N - 1, b - 1, c - 1))
```

<br>

#### min and max

``` min and max
import sys


def init(node, start, end) :
    if start == end :
        tree[node] = [l[start], l[start]]
    else :
        mid = (start + end) // 2
        left, right = init(node * 2, start, mid), init(node * 2 + 1, mid + 1, end)
        tree[node] = [min(left[0], right[0]), max(left[1], right[1])]
    return tree[node]


def search(node, start, end, left, right) :
    if start > right or end < left :
        return [int(1e11), 0]

    if left <= start and end <= right :
        return tree[node]

    mid = (start + end) // 2
    le, ri = search(node * 2, start, mid, left, right), search(node * 2 + 1, mid + 1, end, left, right)
    return [min(le[0], ri[0]), max(le[1], ri[1])]


N, M = map(int, (sys.stdin.readline()).split())
l = []
tree = [0] * (4 * N)
for _ in range(N) :
    l.append(int(sys.stdin.readline().strip()))

init(1, 0, N - 1)
for _ in range(M) :
    a, b = map(int, (sys.stdin.readline()).split())
    print(*search(1, 0, N - 1, a - 1, b - 1))
```

<br>

#### common substring

``` common substring
import sys

A = (sys.stdin.readline().strip())
B = (sys.stdin.readline().strip())

A_len, B_len = len(A), len(B)
dp = [[0 for _ in range(B_len)] for _ in range(A_len)]

max_count = 0

for i in range(A_len) :
    for j in range(B_len) :
        if A[i] == B[j] :
            if i == 0 or j == 0 :
                dp[i][j] = 1
            else :
                dp[i][j] = dp[i - 1][j - 1] + 1
            max_count = max(max_count, dp[i][j])

print(max_count)
```

<br>

#### Oh Keun-soo

```Oh Keun-soo
import sys

N = int(sys.stdin.readline().strip())
A = list(map(int, (sys.stdin.readline()).split()))

result = [-1 for _ in range(N)]
tmp = []
for i in range(N) :
    while tmp and A[tmp[-1]] < A[i] :
        result[tmp.pop()] = A[i]
    tmp.append(i)

print(*result)
```

<br>

#### tree

``` tree
import sys
from collections import deque

N = int(sys.stdin.readline().strip())
parents = list(map(int, (sys.stdin.readline()).split()))
delete_node = int(sys.stdin.readline().strip())

tree, root = {}, 0
for i in range(N) :
    if parents[i] == -1 :
        root = i
    if parents[i] != delete_node and i != delete_node :
        if parents[i] in tree.keys():
            tree[parents[i]].append(i)
        else:
            tree[parents[i]] = [i]


q, count = deque(), 0

if root != delete_node :
    q.append(root)

while q :
    cur = q.popleft()

    if cur in tree.keys() and tree[cur]:
        for i in tree[cur] :
            q.append(i)
    else :
        count += 1
print(count)
```

<br>

#### sort

``` sort
import sys
from collections import deque

N = int(sys.stdin.readline().strip())
numbers = list(map(int, (sys.stdin.readline()).split()))
numbers, number_dict = sorted(numbers), {}
q = deque(numbers)

for i in range(N) :
    target = numbers[i]
    if target in number_dict.keys() :
        number_dict[target] += 1
    else :
        number_dict[target] = 1

while q :
    target = q.popleft()
    if target + 1 in number_dict.keys() :
        state = True
        for j in number_dict.keys() :
            if j > target + 1 :
                state = False
                for _ in range(number_dict[target]) :
                    print(target, end=" ")
                for _ in range(number_dict[target] - 1) :
                    q.popleft()
                print(j, end=" ")
                q.remove(j)
                del number_dict[target]
                if number_dict[j] == 1 :
                    del number_dict[j]
                else :
                    number_dict[j] -= 1
                break
        if state :
            for _ in range(number_dict[target + 1]):
                q.popleft()
            for _ in range(number_dict[target + 1]) :
                print(target + 1, end=" ")
            for _ in range(number_dict[target] - 1):
                q.popleft()
            for _ in range(number_dict[target]):
                print(target, end=" ")
            del number_dict[target], number_dict[target + 1]
    else :
        for _ in range(number_dict[target]) :
            print(target, end=" ")
        for _ in range(number_dict[target] - 1):
            q.popleft()
        del number_dict[target]
```

<br>

#### router

``` router
import sys

N, C = map(int, (sys.stdin.readline()).split())
x, answer = [], 1
for _ in range(N) :
x.append(int(sys.stdin.readline().strip()))
x = sorted(x)


def binary_search(arr, start, end) :
if start > end :
return
mid = (start + end) // 2

current, count = arr[0], 1
for i in range(1, N) :
if arr[i] >= current + mid :
current = arr[i]
count += 1

global answer
if count >= C :
answer = mid
binary_search(arr, mid + 1, end)
else :
answer = mid
binary_search(arr, start, mid - 1)


binary_search(x, 1, x[-1] - x[0])
print(answer)
```

<br>

#### two arr sum

``` two arr sum
import sys, bisect

T = int(sys.stdin.readline().strip())
n = int(sys.stdin.readline().strip())
A = list(map(int, (sys.stdin.readline()).split()))
m = int(sys.stdin.readline().strip())
B = list(map(int, (sys.stdin.readline()).split()))

a_case, b_case = [], []
for i in range(n) :
    result = 0
    for j in range(i, n) :
        result += A[j]
        a_case.append(result)

for i in range(m) :
    result = 0
    for j in range(i, m) :
        result += B[j]
        b_case.append(result)

a_case = sorted(a_case)
total = 0
for i in b_case :
    total += bisect.bisect_right(a_case, T - i) - bisect.bisect_left(a_case, T - i)

print(total)
```

<br>

#### speak mid

``` speak mid
import sys, heapq

N = int(sys.stdin.readline().strip())

left, right, mid = [], [], int(sys.stdin.readline().strip())
left_count, right_count = 0, 0
print(mid)
for i in range(1, N) :
    data = int(sys.stdin.readline().strip())
    if mid > data :
        heapq.heappush(left, -data)
        left_count += 1
    else :
        heapq.heappush(right, data)
        right_count += 1
    if left_count > right_count :
        right_count += 1
        left_count -= 1
        heapq.heappush(right, mid)
        mid = -heapq.heappop(left)
    if right_count > left_count + 1 :
        right_count -= 1
        left_count += 1
        heapq.heappush(left, -mid)
        mid = heapq.heappop(right)
    print(mid)
```

<br>

#### Scale

``` Scale
import sys

N = int(sys.stdin.readline().strip())
arr = sorted(list(map(int, (sys.stdin.readline()).split())))
save_max = arr[0]
if arr[0] != 1 :
    print(1)
else :
    for i in range(1, N) :
        if arr[i] > save_max + 1 :
            break
        else :
            save_max = save_max + arr[i]
    print(save_max + 1)
```

<br>

#### bread store

```bread store
import sys

R, C = map(int, (sys.stdin.readline()).split())
board, count, dx = [], 0, [-1, 0, 1]
for _ in range(R) :
    board.append(list(sys.stdin.readline().strip()))


def dfs(x, y, board) :
    if y == C - 1 :
        return True

    board[x][y] = 'x'
    for k in range(3) :
        mx, my = x + dx[k], y + 1
        if 0 <= mx < R and my < C and board[mx][my] == ".":
            if dfs(mx, my, board) :
                return True
    return False


for i in range(R) :
    if dfs(i, 0, board) :
        count += 1
print(count)
```

<br>

#### Sieve of Eratosthenes

```Sieve of Eratosthenes
import sys


def solution(N, K) :
    count = 0
    states = [True for _ in range(N)]
    for i in range(2, N) :
        if states[i] :
            count += 1
            if count == K :
                return i
            for j in range(i + i, N, i) :
                if states[j] :
                    states[j] = False
                    count += 1
                    if count == K :
                        return j
    return [i for i in range(2, N) if states[i]]


N, K = map(int, (sys.stdin.readline()).split())
print(solution(N + 1, K))
```

<br>

#### Two-dimensional arrays and operations

``` Two-dimensional arrays and operations
import copy, sys

r, c, k = map(int, (sys.stdin.readline()).split())
A = []
c_R, c_C, time = 3, 3, 0
for _ in range(3) :
    A.append(list(map(int, (sys.stdin.readline()).split())))

while time <= 100 :
    if len(A) > r - 1 and len(A[0]) > c - 1 and A[r - 1][c - 1] == k :
        print(time)
        break

    tmp = []
    if c_R >= c_C :
        # R 연산
        max_length = 0
        for i in range(c_R) :
            count_dict, result = {}, []
            for j in range(c_C) :
                if A[i][j] == 0 :
                    continue
                if A[i][j] in count_dict.keys() :
                    count_dict[A[i][j]] += 1
                else :
                    count_dict[A[i][j]] = 1

            key_list = sorted(list(count_dict.keys()), key=lambda x : (count_dict[x], x))
            for key in key_list :
                result.append(key)
                result.append(count_dict[key])

            if len(result) > max_length :
                max_length = len(result)

            tmp.append(result)

        for i in range(c_R) :
            tmp_i_len = len(tmp[i])
            if max_length > tmp_i_len :
                for j in range(max_length - tmp_i_len) :
                    tmp[i].append(0)
        A = copy.deepcopy(tmp)
        c_C = max_length
    else :
        # C 연산
        max_length = 0
        for i in range(c_C):
            count_dict, result = {}, []
            for j in range(c_R):
                if A[j][i] == 0:
                    continue
                if A[j][i] in count_dict.keys():
                    count_dict[A[j][i]] += 1
                else:
                    count_dict[A[j][i]] = 1

            key_list = sorted(list(count_dict.keys()), key=lambda x: (count_dict[x], x))
            for key in key_list:
                result.append(key)
                result.append(count_dict[key])

            if len(result) > max_length:
                max_length = len(result)

            tmp.append(result)

        for i in range(c_C):
            tmp_i_len = len(tmp[i])
            if max_length > tmp_i_len:
                for j in range(max_length - tmp_i_len):
                    tmp[i].append(0)

        trans_tmp = []
        for i in range(max_length) :
            rows = []
            for j in range(len(tmp)) :
                rows.append(tmp[j][i])
            trans_tmp.append(rows)
        A = copy.deepcopy(trans_tmp)
        c_R = max_length
    time += 1

if time > 100 :
    print(-1)
```

<br>

#### Coordinate Compression

``` Coordinate Compression
import sys

N = int(sys.stdin.readline().strip())
A = list(map(int, (sys.stdin.readline()).split()))
A_hat = sorted(list(set(A)))
A_dict = {}
for i in range(len(A_hat)) :
    A_dict[A_hat[i]] = i

for i in range(N) :
    print(A_dict[A[i]], end=" ")
```

<br>

#### Sum of subsequences

``` Sum of subsequences
import sys
from itertools import combinations

N, S = map(int, (sys.stdin.readline()).split())
N_list = list(map(int, (sys.stdin.readline()).split()))

count = 0
for i in range(1, N + 1) :
    for comb in combinations(N_list, i) :
        if S == sum(comb) :
            count += 1
print(count)
```

<br>

#### safe area

``` safe area
import sys
from collections import deque

N = int(sys.stdin.readline().strip())

board = []
for _ in range(N) :
    board.append(list(map(int, (sys.stdin.readline()).split())))

dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]
max_count = 0

for h in range(1, 101) :
    visited = [[True for _ in range(N)] for _ in range(N)]
    count = 0
    for i in range(N) :
        for j in range(N) :
            if visited[i][j] and board[i][j] >= h:
                count += 1
                q = deque()
                q.append([i, j])

                while q :
                    x, y = q.popleft()

                    for k in range(4) :
                        mx, my = x + dx[k], y + dy[k]
                        if 0 <= mx < N and 0 <= my < N and visited[mx][my] and board[mx][my] >= h :
                            visited[mx][my] = False
                            if [mx, my] not in q :
                                q.append([mx, my])

    max_count = max(max_count, count)
print(max_count)
```

<br>

#### Sum of 2D Array

``` Sum of 2D Array
import sys

N, M = map(int, (sys.stdin.readline()).split())
board = []
for _ in range(N) :
    board.append(list(map(int, (sys.stdin.readline()).split())))

sum_board = [[0 for _ in range(M + 1)]]
for i in range(N) :
    tmp = [0]
    for j in range(M) :
        tmp.append(tmp[-1] + board[i][j])
    sum_board.append(tmp)

for i in range(N) :
    for j in range(M + 1) :
        sum_board[i + 1][j] += sum_board[i][j]

K = int(sys.stdin.readline().strip())
for _ in range(K) :
    i, j, x, y = map(int, (sys.stdin.readline()).split())
    print(sum_board[x][y] + sum_board[i - 1][j - 1] - sum_board[x][j - 1] - sum_board[i - 1][y])
```

<br>

#### plus cycle

``` plus cycle
import sys

N = int(sys.stdin.readline().strip())

tmp, count = str(N), 1
if N < 10 :
    tmp += "0"

next_tmp = str(int(tmp[0]) + int(tmp[1]))
if N < 10 :
    tmp = str(N) + next_tmp[-1]
else :
    tmp = tmp[1] + next_tmp[-1]

while int(tmp) != N :
    next_tmp = str(int(tmp[0]) + int(tmp[1]))
    tmp = tmp[1] + next_tmp[-1]

    count += 1
print(count)
```

<br>

#### Castle Defense

``` Castle Defense
import sys, copy
from itertools import combinations
from collections import deque


def attack(attacker, copy_board) :
    remove_list, set_remove = [], set()
    for y in attacker :
        visited = [[True for _ in range(M)] for _ in range(N + 1)]
        visited[N][y] = False

        state = True

        q = deque()
        q.append([N, y, 0])

        while q and state:
            c_x, c_y, dist = q.popleft()

            if dist + 1 > D :
                break

            for k in range(3) :
                mx, my = c_x + dx[k], c_y + dy[k]
                if 0 <= mx < N and 0 <= my < M and visited[mx][my] :
                    if copy_board[mx][my] == 0 :
                        q.append([mx, my, dist + 1])
                        visited[mx][my] = False
                    else :
                        remove_list.append([mx, my])
                        state = False
                        break

            for x, y in remove_list :
                set_remove.add((x, y))
                copy_board[x][y] = 0

            global count
            count += len(set_remove)

            for i in range(N) :
                if sum(copy_board[i]) > 0 :
                    return False
            return True


def move(copy_board) :
    for i in range(N - 1, -1, -1) :
        for j in range(M) :
            if i == N - 1 :
                copy_board[i][j] = 0
            else :
                if copy_board[i][j] == 1 :
                    copy_board[i + 1][j] = 1
                    copy_board[i][j] = 0

    for i in range(N) :
        if sum(copy_board[i]) > 0 :
            return False
    return True


N, M, D = map(int, (sys.stdin.readline()).split())
board, max_count = [], 0
for _ in range(N) :
    board.append(list(map(int, (sys.stdin.readline()).split())))
    board.append([0 for _ in range(M)])

dx, dy = [0, -1, 0], [-1, 0, 1]

for comb in combinations(list(range(M)), 3) :
    count, copy_board = 0, copy.deepcopy(board)
    while True :
        if attack(list(comb), copy_board) :
            break
        if move(copy_board) :
            break
    max_count = max(max_count, count)

print(max_count)
```

<br>

#### find number

```find number
import sys

N = int(sys.stdin.readline().strip())

index = 1
while N > 0 :
    N -= index

    index += 1
result = index + N - 1
if index % 2 == 1 :
    print(str(result) + "/" + str(index - result))
else :
    print(str(index - result) + "/" + str(result))
```

<br>

#### history

``` history
import sys
from collections import deque

n, k = map(int, (sys.stdin.readline()).split())
edges = [[] for _ in range(n + 1)]
pre_events = [set() for _ in range(n + 1)]

for _ in range(k) :
    a, b = map(int, (sys.stdin.readline()).split())
    edges[b].append(a)

for i in range(1, n + 1) :
    q = deque(edges[i])

    for j in edges[i] :
        pre_events[i].add(j)

    visited = [True for _ in range(n + 1)]
    visited[i] = False

    while q :
        node = q.popleft()

        for e in edges[node] :
            if visited[e] :
                visited[e] = False
                pre_events[i].add(e)
                if e not in q :
                    q.append(e)

s = int(sys.stdin.readline().strip())
for _ in range(s) :
    a, b = map(int, (sys.stdin.readline()).split())

    if b in pre_events[a] :
        print(1)
    elif a in pre_events[b] :
        print(-1)
    else :
        print(0)
```

<br>

#### k'th number

``` k'th number
import sys

N = int(sys.stdin.readline().strip())
k = int(sys.stdin.readline().strip())


def binary_search(start, end) :
    if start > end :
        return

    mid = (start + end) // 2

    count = 0
    for i in range(1, N + 1) :
        target = mid // i
        if target > N :
            target = N
        count += target

    if count >= k :
        global result
        result = mid
        binary_search(start, mid - 1)
    else :
        binary_search(mid + 1, end)


result = 0
binary_search(1, min(10 ** 9, N ** 2))
print(result)
```

<br>

#### wormhole

``` wormhole
import sys

TC = int(sys.stdin.readline().strip())
for _ in range(TC) :
    N, M, W = map(int, (sys.stdin.readline()).split())

    edges = [[int(1e9) for _ in range(N + 1)] for _ in range(N + 1)]
    for _ in range(M) :
        S, E, T = map(int, (sys.stdin.readline()).split())
        edges[S][E] = min(edges[S][E], T)
        edges[E][S] = min(edges[E][S], T)

    for _ in range(W) :
        S, E, T = map(int, (sys.stdin.readline()).split())
        edges[S][E] = min(edges[S][E], -T)


    def floyd() :
        for k in range(1, N + 1) :
            for i in range(1, N + 1) :
                for j in range(1, N + 1) :
                    target = edges[i][k] + edges[k][j]
                    if target >= edges[i][j] :
                        continue
                    edges[i][j] = target
                    if target + edges[j][i] < 0 :
                        return True
        return False

    if floyd() :
        print("YES")
    else :
        print("NO")
```

<br>

#### monkey wanting to be a horse

```monkey wanting to be a horse
import sys
from collections import deque

K = int(sys.stdin.readline().strip())
W, H = map(int, (sys.stdin.readline()).split())

board, result = [], -1
for _ in range(H) :
    board.append(list(map(int, (sys.stdin.readline()).split())))

dx, dy = [1, -1, 0, 0, -2, -1, 1, 2, 2, 1, -1, -2], [0, 0, 1, -1, 1, 2, 2, 1, -1, -2, -2, -1]

dp = [[[0 for _ in range(K + 1)] for _ in range(W)] for _ in range(H)]

q = deque()
q.append([0, 0, 0])

while q :
    x, y, k = q.popleft()

    if x == H - 1 and y == W - 1 :
        result = dp[x][y][k]
        break

    for i in range(12) :
        if i > 3 and k >= K :
            continue
        mx, my = x + dx[i], y + dy[i]
        if 0 <= mx < H and 0 <= my < W and board[mx][my] == 0 :
            if i > 3 :
                if dp[mx][my][k + 1] != 0 :
                    continue
                dp[mx][my][k + 1] = dp[x][y][k] + 1
                q.append([mx, my, k + 1])
            else :
                if dp[mx][my][k] != 0 :
                    continue
                dp[mx][my][k] = dp[x][y][k] + 1
                q.append([mx, my, k])

print(result)
```

<br>

#### double priority queue

``` double priority queue
import sys, heapq

T = int(sys.stdin.readline().strip())

for _ in range(T) :
    k = int(sys.stdin.readline().strip())

    exists = [True for _ in range(k + 1)]
    exists[k] = False
    min_q, max_q, in_count = [], [], 0
    for i in range(k) :
        case, value = map(str, (sys.stdin.readline()).split())
        value = int(value)
        if case == "D" :
            if value == 1:
                index = k
                while max_q and not exists[index] :
                    value, index = heapq.heappop(max_q)
                exists[index] = False
            else:
                index = k
                while min_q and not exists[index]:
                    value, index = heapq.heappop(min_q)
                exists[index] = False
        else :
            heapq.heappush(min_q, (value, i))
            heapq.heappush(max_q, (-value, i))
            in_count += 1
    if k - in_count != sum(exists) :
        max_value, min_value = -int(1e20), int(1e20)
        max_index, min_index = k, k
        while max_q and not exists[max_index]:
            max_value, max_index = heapq.heappop(max_q)

        while min_q and not exists[min_index]:
            min_value, min_index = heapq.heappop(min_q)

        print(str(-max_value) + " " + str(min_value))
    else :
        print("EMPTY")
```

<br>

#### palindrome

``` palindrome
import sys

N = int(sys.stdin.readline().strip())
N_list = list(map(int, (sys.stdin.readline()).split()))

dp = [[0 for _ in range(N)] for _ in range(N)]
for i in range(N) :
    dp[i][i] = 1

for i in range(1, N) :
    for j in range(i) :
        # dp[i][j] 판단
        # N_list[j] ~ N_list[i]를 보고 판단
        if N_list[i] == N_list[j]:
            if i - j == 1 :
                dp[i][j] = 1
            elif dp[i - 1][j + 1] == 1 :
                dp[i][j] = 1

M = int(sys.stdin.readline().strip())
for _ in range(M) :
    S, E = map(int, (sys.stdin.readline()).split())
    print(dp[E - 1][S - 1])
```

<br>

#### friend network

``` friend network
import sys


def find(friend_x) :
    if parent[friend_x] != friend_x :
        return find(parent[friend_x])
    return friend_x


def union(friend_a, friend_b) :
    friend_a = find(friend_a)
    friend_b = find(friend_b)

    if friend_a != friend_b :
        if friend_a > friend_b:
            parent[friend_b] = friend_a
            count[friend_a] += count[friend_b]
            print(count[friend_a])
        else:
            parent[friend_a] = friend_b
            count[friend_b] += count[friend_a]
            print(count[friend_b])
    else :
        print(count[friend_a])


T = int(sys.stdin.readline().strip())

for _ in range(T) :
    F = int(sys.stdin.readline().strip())

    parent, count = {}, {}
    for _ in range(F) :
        friend_a, friend_b = map(str, sys.stdin.readline().split())
        if friend_a not in parent.keys() :
            parent[friend_a] = friend_a
            count[friend_a] = 1
        if friend_b not in parent.keys() :
            parent[friend_b] = friend_b
            count[friend_b] = 1

        union(friend_a, friend_b)
```

<br>

#### four integers that sum to zero

```four integers that sum to zero
import sys

n = int(sys.stdin.readline().strip())
A, B, C, D = [], [], [], []
for _ in range(n) :
    a, b, c, d = map(int, (sys.stdin.readline()).split())
    A.append(a)
    B.append(b)
    C.append(c)
    D.append(d)

AB = {}
for a in A :
    for b in B :
        AB[a + b] = AB.get(a + b, 0) + 1

count = 0
for c in C :
    for d in D :
        count += AB.get(-(c + d), 0)

print(count)
```

<br>

#### Minsik Oh's troubles

``` Minsik Oh's troubles
import sys
from collections import deque


def check(node) :
    q = deque()
    q.append(node)
    visited = [True for _ in range(N)]
    visited[node] = False
    while q :
        n = q.popleft()

        for e in edges[n] :
            next_node, cost = e
            if next_node == E :
                return True
            if visited[next_node] :
                visited[next_node] = False
                q.append(next_node)

    return False


def bellman_ford(start) :
    distance[start] = input_cost[S]

    for i in range(N) :
        for j in range(N) :
            for k in edges[j] :
                now_node = j
                next_node, cost = k
                i_cost = input_cost[next_node]

                if distance[now_node] != -int(1e12) and distance[now_node] - cost + i_cost > distance[next_node]:
                    distance[next_node] = distance[now_node] - cost + i_cost
                    if i == N - 1 :
                        if check(next_node) :
                            return True
    return False


N, S, E, M = map(int, (sys.stdin.readline()).split())
edges = [[] for _ in range(N)]
distance = [-int(1e12) for _ in range(N)]

for _ in range(M) :
    start, end, value = map(int, (sys.stdin.readline()).split())
    edges[start].append([end, value])
input_cost = list(map(int, (sys.stdin.readline()).split()))

result = bellman_ford(S)
if distance[E] == -int(1e12) :
    print("gg")
elif result :
    print("Gee")
else :
    print(distance[E])
```

<br>

#### cell phone keyboard

``` cell phone keyboard
import sys


class Node :
    def __init__(self, key, data=None):
        self.key = key
        self.data = data
        self.children = {}
        self.dist = 1


class Trie :
    def __init__(self):
        self.head = Node(None)

    def insert(self, string):
        current_node = self.head
        for char in string :
            if char not in current_node.children :
                current_node.children[char] = Node(char)
            current_node = current_node.children[char]
        current_node.data = string

    def search(self):
        current_node = list(self.head.children.values())
        next_node = []

        while True :
            for node in current_node :
                if node.data :
                    global dist_list
                    dist_list.append(node.dist)

                for c in node.children.values() :
                    c.dist = node.dist

                if len(node.children) > 1 or node.data:
                    for c in node.children.values() :
                        c.dist += 1
                next_node.extend(list(node.children.values()))

            if len(next_node) == 0 :
                break
            else :
                current_node = next_node
                next_node = []


while True :
    try :
        N = int(sys.stdin.readline().strip())
    except :
        break

    trie = Trie()
    for _ in range(N) :
        trie.insert(sys.stdin.readline().strip())

    dist_list = []
    trie.search()
    print("%.2f" % (sum(dist_list) / N))
```

<br>

#### string pow

``` string pow
import sys


def kmp(pattern) :
    pattern_size = len(pattern)
    table = [0 for _ in range(pattern_size)]
    i = 0

    for j in range(1, pattern_size) :
        while i > 0 and pattern[i] != pattern[j] :
            i = table[i - 1]
        if pattern[i] == pattern[j] :
            i += 1
            table[j] = i

    return table


while True :
    data = sys.stdin.readline().strip()
    if data == "." :
        break

    result = kmp(data)
    if len(data) % (len(data) - result[-1]) == 0 :
        print(len(data) // (len(data) - result[-1]))
    else :
        print(1)
```

<br>

#### Kaktus

``` kaktus
import copy, sys
from collections import deque

R, C = map(int, (sys.stdin.readline()).split())
board, waves = [], []
start = [0, 0]
for i in range(R) :
    data = list(sys.stdin.readline().strip())
    board.append(data)
    for j in range(C) :
        if data[j] == 'S' :
            start = [i, j]
        elif data[j] == '*' :
            waves.append([i, j])

dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]

q = deque()
q.append([start[0], start[1], 0])
visited = [[True for _ in range(C)] for _ in range(R)]
visited[start[0]][start[1]] = False

tmp_waves, state, current_dist = [], True, 0
while q and state:
    x, y, dist = q.popleft()

    if dist > current_dist :
        current_dist = dist
        waves = copy.deepcopy(tmp_waves)

    for wx, wy in waves :
        for i in range(4) :
            mx, my = wx + dx[i], wy + dy[i]
            if 0 <= mx < R and 0 <= my < C and board[mx][my] == '.':
                tmp_waves.append([mx, my])
                board[mx][my] = '*'

    for i in range(4) :
        mx, my = x + dx[i], y + dy[i]
        if 0 <= mx < R and 0 <= my < C and visited[mx][my] :
            if board[mx][my] == 'D' :
                print(dist + 1)
                state = False
                break
            elif board[mx][my] == '*' or board[mx][my] == 'X':
                continue
            q.append([mx, my, dist + 1])
            visited[mx][my] = False
if state :
    print("KAKTUS")
```

<br>

#### make bridge

``` make bridge
import sys
from collections import deque

N = int(sys.stdin.readline().strip())
board = []
for _ in range(N) :
    board.append(list(map(int, (sys.stdin.readline()).split())))

dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]

res = 1
tmp_board = [[0 for _ in range(N)] for _ in range(N)]
for i in range(N) :
    for j in range(N) :
        if board[i][j] == 1 and tmp_board[i][j] == 0 :
            q = deque()
            q.append([i, j])
            tmp_board[i][j] = res

            while q :
                x, y = q.popleft()

                for k in range(4) :
                    mx, my = x + dx[k], y + dy[k]
                    if 0 <= mx < N and 0 <= my < N and tmp_board[mx][my] == 0 and board[mx][my] == 1:
                        if [mx, my] not in q :
                            q.append([mx, my])
                        tmp_board[mx][my] = res
            res += 1

min_dist = int(1e9)
for i in range(1, res) :
    for j in range(N) :
        for k in range(N) :
            if tmp_board[j][k] == i :
                q = deque()
                q.append([j, k, 0])

                visited = [[True for _ in range(N)] for _ in range(N)]
                visited[j][k] = False

                state = True
                while q and state:
                    x, y, dist = q.popleft()

                    for d in range(4) :
                        mx, my = x + dx[d], y + dy[d]
                        if 0 <= mx < N and 0 <= my < N and tmp_board[mx][my] != i and visited[mx][my]:
                            if board[mx][my] != 0 :
                                min_dist = min(min_dist, dist)
                                state = False
                                break
                            q.append([mx, my, dist + 1])
                            visited[mx][my] = False

print(min_dist)
```

<br>

#### coffee shop2

``` coffee shop2
import sys


def init(node, start, end) :
    if start == end :
        tree[node] = N_list[start]
    else:
        mid = (start + end) // 2
        tree[node] = init(node * 2, start, mid) + init(node * 2 + 1, mid + 1, end)
    return tree[node]


def subSum(node, start, end, left, right) :
    if start > right or end < left :
        return 0

    if left <= start and end <= right :
        return tree[node]

    mid = (start + end) // 2
    return subSum(node * 2, start, mid, left, right) + subSum(node * 2 + 1, mid + 1, end, left, right)


def update(node, start, end, index, diff) :
    if index < start or index > end :
        return

    tree[node] += diff
    if start != end :
        mid = (start + end) // 2
        update(node * 2, start, mid, index, diff)
        update(node * 2 + 1, mid + 1, end, index, diff)


N, Q = map(int, (sys.stdin.readline()).split())
N_list = list(map(int, (sys.stdin.readline()).split()))

tree = [0] * (N * 5)
init(1, 0, N - 1)

for _ in range(Q) :
    x, y, a, b = map(int, (sys.stdin.readline()).split())
    if x > y :
        print(subSum(1, 0, N - 1, y - 1, x - 1))
    else :
        print(subSum(1, 0, N - 1, x - 1, y - 1))

    a = a - 1
    diff = b - N_list[a]
    N_list[a] = b
    update(1, 0, N - 1, a, diff)
```

<br>

#### The one in the green suit is Zelda, right?

``` The one in the green suit is Zelda, right?
import java.util.PriorityQueue;
import java.util.Scanner;

class Point implements Comparable<Point>{
    private int x;
    private int y;
    private int cost;

    public Point(int x, int y, int cost){
        this.x = x;
        this.y = y;
        this.cost = cost;
    }

    public int getX(){
        return this.x;
    }

    public int getY(){
        return this.y;
    }

    public int getCost(){
        return this.cost;
    }

    @Override
    public int compareTo(Point point) {
        if(this.cost >= point.cost){
            return 1;
        }
        return -1;
    }
}

public class Main {
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);
        int count = 1;
        while(true){
            int N = scan.nextInt();
            if(N == 0){break;}
            int board[][] = new int[N][N];
            int dp[][] = new int[N][N];
            int dx[] = {1, -1, 0, 0};
            int dy[] = {0, 0, 1, -1};

            for(int i=0;i<N;i++){
                for(int j=0;j<N;j++){
                    dp[i][j] = Integer.MAX_VALUE;
                    board[i][j] = scan.nextInt();
                }
            }

            PriorityQueue<Point> priorityQueue = new PriorityQueue<>();
            priorityQueue.add(new Point(0, 0, board[0][0]));
            dp[0][0] = board[0][0];


            while(!priorityQueue.isEmpty()){
                Point point = priorityQueue.poll();
                int x = point.getX();
                int y = point.getY();
                int cost = point.getCost();

                if(x == N - 1 && y == N - 1){
                    continue;
                }

                for(int i=0;i<4;i++){
                    int mx = point.getX() + dx[i];
                    int my = point.getY() + dy[i];
                    if((0<=mx && mx<N) && (0<=my && my<N)){
                        int cost_result = cost + board[mx][my];
                        if(dp[mx][my] > cost_result) {
                            dp[mx][my] = cost_result;
                            priorityQueue.add(new Point(mx, my, cost_result));
                        }
                    }
                }
            }

            System.out.println("Problem " + count + ": " + dp[N - 1][N - 1]);
            count++;
        }
    }
}
```

<br>

#### IOIOI

``` IOIOI
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Stack;

public class Main {

    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        int N = Integer.parseInt(br.readLine());
        int M = Integer.parseInt(br.readLine());

        char[] list = br.readLine().toCharArray();
        int count = 0;
        int size = 0;
        Stack<Character> stack = new Stack<>();
        stack.add(list[0]);
        for(int i=1;i<M;i++){
            if(stack.isEmpty() || stack.get(size) != list[i]){
                stack.add(list[i]);
                size++;
                if(list[i] == 'I' && size + 1 >= (N * 2) + 1){
                    count += 1;
                }
            }
            else {
                stack.clear();
                stack.add(list[i]);
                size = 0;
            }
        }
        System.out.println(count);
    }
}
```

<br>

#### road pavement

```road pavement
import sys, heapq

N, M, K = map(int, (sys.stdin.readline()).split())

edges = [[] for _ in range(N + 1)]
for _ in range(M) :
    a, b, c = map(int, (sys.stdin.readline()).split())
    edges[a].append([b, c])
    edges[b].append([a, c])

distances = [[int(1e12) for _ in range(N + 1)] for _ in range(K + 1)]
distances[0][1] = 0
q = [[0, 1, 0]]
while q :
    dist, current_node, wall_break = heapq.heappop(q)

    if current_node == N or dist > distances[wall_break][current_node]:
        continue

    for i in edges[current_node] :
        cost = dist + i[1]
        if distances[wall_break][i[0]] > cost :
            distances[wall_break][i[0]] = cost
            heapq.heappush(q, [cost, i[0], wall_break])
        if K > wall_break and distances[wall_break + 1][i[0]] > dist:
            distances[wall_break + 1][i[0]] = dist
            heapq.heappush(q, [dist, i[0], wall_break + 1])

result = int(1e12)
for i in range(K + 1) :
    result = min(result, distances[i][N])
print(result)
```

<br>

#### LCA 2

``` LCA 2
import sys
from collections import deque

N = int(sys.stdin.readline().strip())
LENGTH = 21
graph = [[] for _ in range(N + 1)]
parent = [[0 for _ in range(LENGTH)] for _ in range(N + 1)]
visited = [True for _ in range(N + 1)]
depth = [0 for _ in range(N + 1)]

for _ in range(N - 1) :
    a, b = map(int, (sys.stdin.readline()).split())
    graph[a].append(b)
    graph[b].append(a)

q = deque()
q.append([1, 0])
depth[1], visited[1] = 0, False

while q :
    node, dist = q.popleft()

    for i in graph[node] :
        if visited[i]:
            parent[i][0], visited[i] = node, False
            depth[i] = dist + 1
            q.append([i, dist + 1])

for i in range(1, LENGTH):
    for j in range(1, N + 1):
        parent[j][i] = parent[parent[j][i - 1]][i - 1]

M = int(sys.stdin.readline().strip())
for _ in range(M) :
    a, b = map(int, (sys.stdin.readline()).split())

    if depth[a] > depth[b] :
        a, b = b, a

    for i in range(LENGTH - 1, -1, -1):
        if depth[b] - depth[a] >= 2 ** i:
            b = parent[b][i]

    if a == b :
        print(a)
        continue

    for i in range(LENGTH - 1, -1, -1):
        if parent[a][i] != parent[b][i]:
            a = parent[a][i]
            b = parent[b][i]

    print(parent[a][0])
```

<br>

#### planet tunnel

``` planet tunnel
import sys, heapq


def find_parent(x, parent) :
    if parent[x] != x :
        return find_parent(parent[x], parent)
    return x


def union_parent(a, b, parent) :
    a = find_parent(a, parent)
    b = find_parent(b, parent)

    if a > b :
        parent[a] = b
    else :
        parent[b] = a


N = int(sys.stdin.readline().strip())

edges = []
storage = []
parent = [i for i in range(N + 1)]
for i in range(1, N + 1) :
    a, b, c = map(int, (sys.stdin.readline()).split())
    storage.append([a, b, c, i])

storage = sorted(storage, key=lambda x : x[0])
for i in range(N - 1) :
    heapq.heappush(edges, [abs(storage[i + 1][0] - storage[i][0]), storage[i + 1][3], storage[i][3]])

storage = sorted(storage, key=lambda x : x[1])
for i in range(N - 1) :
    heapq.heappush(edges, [abs(storage[i + 1][1] - storage[i][1]), storage[i + 1][3], storage[i][3]])

storage = sorted(storage, key=lambda x : x[2])
for i in range(N - 1) :
    heapq.heappush(edges, [abs(storage[i + 1][2] - storage[i][2]), storage[i + 1][3], storage[i][3]])

result, count = 0, 0
while count != N - 1 :
    value, node_1, node_2 = heapq.heappop(edges)
    if find_parent(node_1, parent) != find_parent(node_2, parent) :
        union_parent(node_1, node_2, parent)
        result += value
        count += 1

print(result)
```

<br>

#### high rise

``` high rise
import sys

N = int(sys.stdin.readline().strip())
heights = list(map(int, (sys.stdin.readline()).split()))

max_count = 0
for i in range(N) :
    count = 0
    s_x, s_y = i, heights[i]

    for j in range(i - 1, -1, -1) :
        m_x, m_y = j, heights[j]
        state = True
        for k in range(i - 1, j, -1) :
            a = (s_y - m_y) / (s_x - m_x)
            if heights[k] >= s_y - (i - k) * a:
                state = False
                break
        if state :
            count += 1
    for j in range(i + 1, N, 1) :
        m_x, m_y = j, heights[j]
        state = True
        for k in range(i + 1, j, 1):
            a = (s_y - m_y) / (s_x - m_x)
            if heights[k] >= s_y + (k - i) * a:
                state = False
                break
        if state:
            count += 1

    max_count = max(max_count, count)

print(max_count)
```
