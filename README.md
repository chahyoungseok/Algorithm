# Algorithm

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#greedy">Greedy</a></li>
    <li><a href="#avatar">Avatar</a></li>
    <li><a href="#dfs-bfs">DFS BFS</a></li>
    <li><a href="#sort">Sort</a></li>
    <li><a href="#search">Search</a></li>
    <li><a href="#dynamic-programming">Dynamic Programming</a></li>
    <li><a href="#shortest-path">Shortest Path</a></li>
    <li><a href="#graph">Graph</a></li>
    <li><a href="#practice-solution">Practice Solution</a></li>
    <li><a href="#programmers">Programmers</a></li>
  </ol>
</details>
<br>

## Greedy
 

### 거스름돈

``` changeMoney
changeMoney = int(input("거스름돈을 입력하세요: "))
money_unit = [500, 100, 50, 10]

for i in range(0,4) :
    print(str(money_unit[i]) + "원의 개수는 : " + str(int(changeMoney / money_unit[i])) + "개 입니다.")
    changeMoney %= money_unit[i]
```

<br>

### 큰수의 법칙

``` law of large numbers 
// N개 주어진 자연수들을 연속으로 더하되 K번 초과하지않고 총 M번 더했을 때의 최대값 (중복가능)

N, M, K = map(int,input().split())

intArr = list(map(int,input().split()))
intArr.sort()

max = intArr[N - 1]
submax = intArr[N - 2]

count = int(M/(K + 1))

sum = submax * count + max * (M - count)
print(sum)
``` 

<br>

### 숫자 카드 게임

``` number card game
// 각각의 행에 해당하는 M개의 자연수들 중 최솟값을 구하고, 그 N개의 열중 최댓값을 구하여라

N, M = map(int, input().split()) // N은 행의 개수, M은 열의 개수

card = []

for inList in range(0,N) :
    inList = list(map(int,input().split()))
    card.append(min(inList))

print(max(card))
```

<br>

### 1이 될 때까지

``` until it becomes 1
// 1. N에서 1을 뺀다.
// 2. N을 K로 나눈다. (나누어 떨어질때만)
// 위의 1 ,2를 1이 될 때까지 수행하는 최소연산의 수는?

N, K = map(int, input().split())

count = 0

while True :
    remainder = (N // K) * K
    count += N - remainder
    N = remainder

    if N < K :
        break

    count += 1
    N //= K

count += N - 1

print(count)
```

<br>

## Avatar


### 상하좌우

``` Up Down Left Right
// 여행가가 L, R, U, D으로 움직이고 정해진 N x N 지도에서 벗어나게되면 무시하는 방법으로 여행가의 마지막위치를 추적하는 프로그램
N = int(input())
moveList = list(map(str, input().split()))

x, y = 1, 1
preX, preY = 0, 0

move_types = ['L','R','U','D']

mx = [0,0,-1,1]
my = [-1,1,0,0]

for move in moveList :
    for i in range(0,len(move_types)) :
        if move == move_types[i] :
            preX = x + mx[i]
            preY = y + my[i]
            break

    if (preX > 0 and preX < 6) and (preY > 0 and preY < 6) :
        x = preX
        y = preY

print(str(x) + " " + str(y))
```
 
<br>

### 시각

``` clock
N = int(input())

count = 0
for i in range(0, N + 1) :
    for j in range(0, 60) :
        for h in range(0, 60) :
            if '3' in str(i) + str(j) + str(h) :
                count += 1

print(count)
```

<br>

### 왕실의 나이트

``` royal night
def numberOfCase(canMove, RC) :
    if RC == 1 or RC == 8:
        canMove /= 2
    elif RC == 2 or RC == 7:
        if canMove % 4 == 0:
            canMove *= 3 / 4
        else:
            canMove *= 2 / 3

    return canMove

loc = input()

row = int(loc[1])
col = ord(loc[0]) - ord('a') + 1

canMove = 8
canMove = numberOfCase(canMove,row)
canMove = numberOfCase(canMove,col)

print(int(canMove))
```

<br>

### 게임 개발

``` game development
N, M = map(int, input().split())
a, b, d = map(int, input().split())

mapVis = [[0] * M for i in range(N)]
mapVis[a][b] = 1
visCount = 1

mapInf = []
for i in range(N) :
    mapInf.append(list(map(int, input().split())))

dirCount = 0
dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]

while True :
    d = (d + 3) % 4
    CooA = a + dx[d]
    CooB = b + dy[d]

    if dirCount == 4 :
        invD = (d + 2) % 4
        a = a + dx[invD]
        b = b + dy[invD]

        if mapVis[a][b] == 1 or mapInf[a][b] == 1 :
            break
        else :
            mapVis[a][b] = 1
            visCount += 1
            continue

    if CooA < 0 or CooA > N - 1 or mapVis[CooA][CooB] == 1 or mapInf[CooA][CooB] == 1 :
        dirCount += 1
        continue
    else :
        a = CooA
        b = CooB
        mapVis[a][b] = 1
        visCount += 1
        dirCount = 0

print(visCount)
```

<br>

## DFS BFS


### 이론

탐색
 - 많은 양의 데이터 중에서 원하는 데이터를 찾는 과정

자료구조
 - 데이터를 표현하고 관리하고 처리하기 위한 구조

Stack 사용법 (in Python)
 - 별도의 라이브러리가 필요없고, 기본 리스트에서 append, pop등을 사용

Queue 사용법 (in Python)
 - from collections import deque
 - deque : stack과 queue의 장점을 모두 채택한 것

재귀함수
 - 자기 자신을 다시 호출하는 함수

<br><br>

DFS
 - 깊이우선탐색 이라고 부르며, 그래프에서 깊은 부분을 우선적으로 탐색하는 알고리즘

``` dfs
def dfs(graph,  v, visited) :
    visited[v] = True
    print(v, end=' ')

    for i in graph[v] :
        if not visited[i] :
            dfs(graph, i, visited)

graph =[
    [],
    [2,3,8],
    [1,7],
    [1,4,5],
    [3,5],
    [3,4],
    [7],
    [2,6,8],
    [1,7]
]

visited = [False] * 9

dfs(graph, 1, visited)
```

<br><br>

BFS
 - 너비우선탐색 이라고 부르며, 가까운 노드부터 탐색하는 알고리즘
 - 인접한 노드가 여러개 있을 때, 숫자가 작은 노드부터 큐에 삽입하였다

``` bfs
from collections import deque

def bfs (graph, start, visited) :
    queue = deque([start])

    visited[start] = True

    while queue :
        v = queue.popleft()
        print(v, end=' ')

        for i in graph[v] :
            if not visited[i]:
                queue.append(i)
                visited[i] = True

graph =[
    [],
    [2,3,8],
    [1,7],
    [1,4,5],
    [3,5],
    [3,4],
    [7],
    [2,6,8],
    [1,7]
]

visited = [False] * 9

bfs(graph, 1, visited)
```

<br>

### 음료수 얼리기

``` freeze drinks
def confirmAround(i, j) :
    global iceMap

    if i < 0 or i > len(iceMap) - 1 or j < 0 or j > len(iceMap[0]) - 1  :
        return False

    if iceMap[i][j] == 0 :
        iceMap[i][j] = 1

        confirmAround(i - 1, j)
        confirmAround(i, j + 1)
        confirmAround(i + 1, j)
        confirmAround(i, j - 1)

        return True
    return False


N, M = map(int, input().split())

iceMap = []
iceCount = 0

for i in range(N) :
    iceMap.append(list(map(int,input())))


for i in range(N) :
    for j in range(M) :
        if confirmAround(i, j) :
            iceCount += 1

print(iceCount)
```

<br>

### 미로 탈출

``` Escape maze
from collections import deque

N, M = map(int, input().split())

mazeMap = []
for i in range(N) :
    mazeMap.append(list(map(int, input())))

x, y = 0, 0

dx = [-1, 0, 1, 0]
dy = [0, -1, 0, 1]

mazeQueue = deque()
mazeQueue.append((x, y))

while mazeQueue :
    x, y = mazeQueue.popleft()

    for i in range(4) :
        preX = dx[i] + x
        preY = dy[i] + y

        if preX < 0 or preY < 0 or preX > N - 1 or preY > M - 1 :
            continue

        if mazeMap[preX][preY] == 1 :
            mazeQueue.append((preX, preY))
            mazeMap[preX][preY] = mazeMap[x][y] + 1

print(mazeMap[N - 1][M - 1])
```

<br>


## Sort


### 선택정렬

``` select sort
array = [7,5,9,0,3,1,6,2,4,8]

for i in range(len(array)) :
    min_index = i
    for j in range(i+1, len(array)) :
        if array[min_index] > array[j] :
            min_index = j
    array[i], array[min_index] = array[min_index], array[i]

print(array)
```

<br>

### 삽입정렬

``` insert sort
array = [7,5,9,0,3,1,6,2,4,8]

for i in range(1, len(array)) :
    for j in range(i, 0, -1) :
        if array[j] < array[j - 1] :
            array[j], array[j - 1] = array[j - 1], array[j]
        else :
            break

print(array)
```

<br>

### 퀵 정렬

``` quick sort
array = [5,7,9,0,3,1,6,2,4,8]

def quick_sort(array, start, end) :
    if start >= end :
        return

    pivot = start
    left = start + 1
    right = end

    while left <= right :
        while left <= end and array[left] <= array[pivot] :
            left += 1

        while right > start and array[right] >= array[pivot] :
            right -= 1

        if left > right :
            array[right], array[pivot] = array[pivot], array[right]
        else :
            array[left], array[right] = array[right], array[left]

    quick_sort(array, start, right - 1)
    quick_sort(array, right + 1, end)

quick_sort(array, 0, len(array) - 1)
print(array)
```

<br>

### 계수 정렬

<br>

특징
 - 데이터의 개수가 1000개정도 이하일 때 사용하면 너무좋은 알고리즘

``` count sort
array = [7,5,9,0,3,1,6,2,9,1,4,8,0,5,2]

count = [0] * (max(array) + 1)

for i in range(len(array)) :
    count[array[i]] += 1

for i in range(len(count)) :
    for j in range(count[i]) :
        print(i, end=' ')
```

<br>

### 큰수의 법칙

``` law of large numbers 
// N개 주어진 자연수들을 연속으로 더하되 K번 초과하지않고 총 M번 더했을 때의 최대값 (중복가능)

N, M, K = map(int,input().split())

intArr = list(map(int,input().split()))
intArr.sort()

max = intArr[N - 1]
submax = intArr[N - 2]

count = int(M/(K + 1))

sum = submax * count + max * (M - count)
print(sum)
```

<br>

### 위에서 아래로

``` top down
N = int(input())

array = []
for i in range(N) :
    array.append(int(input()))

array = sorted(array, reverse=True)

for i in array :
    print(i, end=' ')
```

<br>

### 성적이 낮은 순서로 학생 출력하기

``` print students in descending order of grades
N = int(input())

array = []

for i in range(N) :
    input_data = input().split()
    array.append((input_data[0], int(input_data[1])))

array = sorted(array, key= lambda score: score[1])

for student in array :
    print(student[0], end=' ')
```

<br>

### 두 배열의 원소 교체

``` swap elements in two arrays
N, K = map(int, input().split())

A = list(map(int, input().split()))
B = list(map(int, input().split()))

A.sort()
B.sort(reverse=True)

for i in range(K) :
    if B[i] > A[i] :
        A[i], B[i] = B[i], A[i]
    else :
        break

print(sum(A))
```

<br>

## Search


### 순차탐색

``` sequential search
def sequential_search(n, target, array) :
    for i in range(n) :
        if array[i] == target :
            return i + 1

print("생성할 원소 개수를 입력한 다음 한 칸 띄고 찾을 문자열을 입력하세요.")
input_data = input().split()
n = int(input_data[0])
target = input_data[1]

print("앞서 적은 원소 개수만큼 문자열을 입력하세요. 구분은 띄어쓰기 한 칸으로 합니다.")
array = input().split()

print(sequential_search(n, target, array))
```

<br>

### 이진탐색

``` binary search
# 데이터가 정렬되어 있을 때, 사용할 수 있다.
# 시간 복잡도가 O(logN) 이기 떄문에 데이터가 1000만 이상으로 가게되면 가급적 사용해야한다.

def binary_search(array, target, start, end) :
    if start > end:
        return None
    mid = (start + end) // 2

    if array[mid] == target :
        return mid
    elif array[mid] > target :
        return binary_search(array, target, start, mid - 1)
    else :
        return binary_search(array, target, mid + 1, end)

n, target = list(map(int, input().split()))

array = list(map(int, input().split()))

result = binary_search(array, target, 0, n - 1)
if result == None :
    print("원소가 존재하지 않습니다.")
else :
    print(result + 1)
```

<br>

### sys 라이브러리

``` sys
# 데이터의 개수가 1000만을 넘어가거나 탐색범위의 크기가 1000억 이상인 문제들이 나올 때, input()은 동작속도가 느리므로 아래의 라이브러리를 응용하자.

import sys

input_data = sys.stdin.readline().rstrip()

print(input_data)
```

<br>

### 부품 찾기

``` find equipments
def binary_search(array, target, start, end) :
    if start > end:
        return None
    mid = (start + end) // 2

    if array[mid] == target :
        return mid
    elif array[mid] > target :
        return binary_search(array, target, start, mid - 1)
    else :
        return binary_search(array, target, mid + 1, end)


N = int(input())
equipments_shop = list(map(int, input().split()))
equipments_shop.sort()

M = int(input())
equipments_customer = list(map(int, input().split()))

for i in equipments_customer :
    result = binary_search(equipments_shop, i, 0, N - 1)

    if result != None :
        print("yes", end =' ')
    else :
        print("no", end = ' ')
```

<br>

### 떡볶이 떡 만들기

``` make rice cake
N, M = map(int, input().split())
riceCake = list(map(int, input().split()))

result = 0
start = 0
end = max(riceCake)

while start <= end :
    remainCake = 0
    mid = (start + end) // 2

    for i in riceCake :
        if i > mid :
            remainCake += i - mid

    if remainCake > M :
        result = mid
        start = mid + 1
    elif remainCake == M :
        result = mid
        break
    else :
        end = mid - 1

print(result)
```

<br>

## Dynamic Programming


### 이론

다이나믹 프로그래밍
 - 큰 문제를 작게 나누고, 같은 문제라면 한 번씩만 풀어 문제를 효율적으로 해결하는 알고리즘 기법.

메모제이션 or 캐싱
 - 한 번 구한 결과를 메모리 공간에 메모해두고 같은 식을 다시 호출하면 메모한 결과를 그대로 가져오는 기법.


<br>

### Top Down 방식(하향식)
 - 메모제이션 방식을 사용하지 않음.
 - 구현가능하다면 Bottom Up 방식을 사용하는게 좋음.

``` top down
d = [0] * 100

def pibo(x) :
    print("f(" + str(x) + ")", end=" ")
    if x == 1 or x == 2 :
        return 1
    if d[x] != 0 :
        return d[x]
    d[x] = pibo(x - 1) + pibo(x - 2)
    return d[x]

pibo(6)
```

### Bottom Up 방식(상향식)
 - 메모제이션 방식을 사용함.

``` bottom up
d = [0] * 100

d[1] = 1
d[2] = 1
n = 99

for i in range(3, n + 1) :
    d[i] = d[i - 1] + d[i - 2]

print(d[n])
```

<br>

### 1로 만들기

``` make 1
X = int(input())

d = [0] * 30001

for i in range(2, X + 1):
    d[i] = d[i - 1] + 1

    if i % 2 == 0 :
        d[i] = min(d[i], d[i // 2] + 1)

    if i % 3 == 0:
        d[i] = min(d[i], d[i // 3] + 1)

    if i % 5 == 0:
        d[i] = min(d[i], d[i // 5] + 1)

print(d[X])
```

<br>

### 개미 전사

``` warrier ant
N = int(input())
storage = list(map(int, input().split()))

d = [0] * 100

d[0] = storage[0]
d[1] = max(storage[0], storage[1])

for i in range(2, N) :
    d[i] = max(d[i - 1], storage[i] + d[i - 2])

print(d[N-1])
```

<br>

### 효율적인 화폐 구성

``` efficient money structure
N, M = map(int, input().split())

moneyAry = []
for i in range(N) :
    moneyAry.append(int(input()))

d = [10001] * (M+1)
d[0] = 0

for i in range(N) :
    for j in range(moneyAry[i],M + 1) :
        if d[j - moneyAry[i]] != 10001 :
            d[j] = min(d[j],d[j - moneyAry[i]] + 1)

if d[M] == 10001 :
    print(-1)
else :
    print(d[M])
```

<br>

## Shortest Path


### 이론


Dijkstra
 - 음의 간선이 없을때, 정상동작
 - 그리디 알고리즘으로 분류됨
 - 각 노드들에 대한 최단거리 정보를 항상 1차원리스트에 저장


<br>

### Dijkstra

``` dijkstra
import heapq

INF = int(1e9)

n, m = map(int, input().split())
start = int(input())

graph = [[] for i in range(n + 1)]
distance = [INF] * (n + 1)

for _ in range(m) :
    a, b, c = map(int, input().split())
    graph[a].append((b,c))

# graph는 (노드, 거리) / Heap은 (거리, 노드) 를 주의!
def dijkstra(start) :
    q = []
    heapq.heappush(q,(0,start))
    distance[start] = 0

    while q:
        dist, now = heapq.heappop(q)
        if distance[now] < dist :
            continue

        for i in graph[now] :
            cost = dist + i[1]
            if cost < distance[i[0]] :
                distance[i[0]] = cost
                heapq.heappush(q,(cost, i[0]))

dijkstra(start)

for i in range(1, n + 1) :
    if distance[i] == INF :
        print("INFINITY")
    else :
        print(distance[i])
```

Floyd
 - 모든 지점에서 다른 모든 지점까지의 최단경로
 - 다이나믹 프로그래밍으로 분류된다.


<br>

### Floyd

``` floyd
n, m = map(int, input().split())

graph = [[int(1e9)] * (n + 1) for _ in range(n + 1)]

for i in range(1, n+1) :
    graph[i][i] = 0

for _ in range(m) :
    a, b, c = map(int, input().split())
    graph[a][b] = c

for k in range(1, n+1) :
    for a in range(1, n+1) :
        for b in range(1, n+1) :
            graph[a][b] = min(graph[a][b], graph[a][k] + graph[k][b])

for a in range(1, n+1) :
    for b in range(1, n+1) :
        if graph[a][b] == int(1e9) :
            print("INFINITY", end=" ")
        else :
            print(graph[a][b], end=" ")
    print()
```

<br>

### 미래 도시

``` future city
N, M = map(int, input().split())

INF = int(1e9)
graph = [[INF] * (N + 1) for _ in range(N + 1)]

for i in range(1, N + 1) :
    graph[i][i] = 0

for i in range(M) :
    a,b = map(int, input().split())
    graph[a][b] = 1
    graph[b][a] = 1

X, K = map(int, input().split())

for k in range(1,N + 1) :
    for i in range(1, N + 1) :
        for j in range(1, N + 1) :
            graph[i][j] = min(graph[i][k] + graph[k][j], graph[i][j])

distances = graph[1][K] + graph[K][X]

if distances >= INF :
    print("-1")
else :
    print(distances)
```

<br>

### 전보

``` telegram
import heapq

INF = int(1e9)
visitCount = 0
maxDistance = 0

N, M, C = map(int, input().split())

graph = [[] for _ in range(N + 1)]

for i in range(M) :
    X, Y, Z = map(int, input().split())
    graph[X].append((Y,Z))

distances = [INF for _ in range(N + 1)]

q = []
heapq.heappush(q, (0,C))
distances[C] = 0

while q :
    dist, now = heapq.heappop(q)

    if dist > distances[now] :
        continue

    for i in graph[now] :
        cost = dist + i[1]
        if distances[i[0]] > cost :
            distances[i[0]] = cost
            heapq.heappush(q, (cost, i[0]))

for i in range(1, M + 1) :
    if distances[i] != INF :
        visitCount += 1
        maxDistance = max(maxDistance, distances[i])

print("가장 멀리있는 노드의 거리는 : " + str(maxDistance) + "\n방문한 노드의 개수는 : " + str(visitCount))
```

<br>

## Graph


### 이론

인접행렬 
 - 2차원 배열을 사용하는 방식
 - ex) Floyd

인접리스트
 - 리스트를 사용하는 방식
 - Dijkstra

서로소 집합
 - 공통원소가 없는 두 집합을 의미한다.
 - 서로소 부분 집합들로 나누어진 원소들의 데이터를 처리하기 위한 자료구조로 서로소 집합 자료구조가 있다.
 - cf) union - find 자료구조

<br>

### union - find 
 - 1. union 연산을 확인하여, 서로 연결된 두 노드 A,B를 확인한다. 
 - 2. A와 B의 루트노드를 각각 찾고, 작은 루트노드를 큰 루트노드가 가르키게한다.
 - 3. 모든 union 연산을 처리할 때까지 1,2번과정을 반복한다.

```union - find
def find_parent(parent, x) :
    if parent[x] != x:
        return find_parent(parent, parent[x])
    return parent[x]

def union_parent(parent, a, b) :
    a = find_parent(parent, a)
    b = find_parent(parent, b)

    if a < b :
        parent[b] = a
    else :
        parent[a] = b

v, e = map(int, input().split())
parent = [0] * (v + 1)

for i in range(1, v + 1) :
    parent[i] = i

for i in range(e) :
    a, b = map(int, input().split())
    union_parent(parent, a, b)

print("각 원소가 속한 집합: ", end='')
for i in range(1, v + 1) :
    print(find_parent(parent, i), end=' ')

print()

print("부모 테이블: ", end='')
for i in range(1, v + 1) :
    print(parent[i], end=' ')
```

<br>

### 사이클 판별 알고리즘

``` discriminate cycle
def find_parent(parent, x) :
    if parent[x] != x:
        return find_parent(parent, parent[x])
    return parent[x]

def union_parent(parent, a, b) :
    a = find_parent(parent, a)
    b = find_parent(parent, b)

    if a < b :
        parent[b] = a
    else :
        parent[a] = b


v, e = map(int, input().split()) 
parent = [0] * (v + 1)

for i in range(1, v + 1) :
    parent[i] = i

cycle = False

for i in range(e) :
    a, b = map(int, input().split())
    if find_parent(parent, a) == find_parent(parent, b) :
        cycle = True
        break
    else :
        union_parent(parent, a, b)

if cycle :
    print("사이클이 발생했습니다.")
else :
    print("사이클이 발생하지 않았습니다.")
```

<br>

### Kruskal 알고리즘

``` kruskal
def find_parent(parent, x) :
    if parent[x] != x:
        return find_parent(parent, parent[x])
    return parent[x]

def union_parent(parent, a, b) :
    a = find_parent(parent, a)
    b = find_parent(parent, b)

    if a < b :
        parent[b] = a
    else :
        parent[a] = b


v, e = map(int, input().split())
parent = [0] * (v + 1)

edges = []
result = 0

for i in range(1, v + 1) :
    parent[i] = i

cycle = False

for i in range(e) :
    a, b, cost = map(int, input().split())
    edges.append((cost, a, b))

edges.sort()

for edge in edges :
    cost, a, b = edge
    if find_parent(parent, a) != find_parent(parent, b) :
        union_parent(parent, a, b)
        result += cost

print(result)
```

<br>

### topological sort
 - 1. 진입차수가 0인 노드를 큐에 넣는다.
 - 2. 큐에서 원소를 꺼내 해당 노드에서 출발하는 간선을 그래프에서 제거한다.
 - 3. 새롭게 진입차수가 0이 된 노드를 큐에 넣는다.
 - 4. 2번과 3번의 과정을 큐가 빌 때까지 반복한다.

``` topological sort
from collections import deque

v, e = map(int, input().split())

indegree = [0] * (v + 1)

graph = [[] for i in range(v + 1)]

for _ in range(e):
    a, b = map(int, input().split())
    graph[a].append(b)

    indegree[b] += 1


def topology_sort():
    result = []
    q = deque()

    for i in range(1, v + 1):
        if indegree[i] == 0:
            q.append(i)

    while q:
        now = q.popleft()
        result.append(now)

        for i in graph[now]:
            indegree[i] -= 1
            if indegree[i] == 0:
                q.append(i)

    for i in result:
        print(i, end=' ')


topology_sort()
```

<br>

#### 팀 결성

``` make team
def find_parent(parent, x) :
    if parent[x] != x :
        parent[x] = find_parent(parent, parent[x])
    return parent[x]

def union_parent(parent, a, b) :
    a = find_parent(parent, a)
    b = find_parent(parent, b)
    if a > b :
        parent[a] = b
    else :
        parent[b] = a

N, M = map(int, input().split())
parent = [0] * (N + 1)

for i in range(1, N) :
    parent[i] = i

for _ in range(M) :
    w, a, b = map(int, input().split())
    if w == 0 :
        union_parent(parent, a, b)
    else :
        if find_parent(parent, a) == find_parent(parent, b) :
            print("Yes")
        else :
            print("No")
```

<br>

#### 도시 분할 계획

``` division city plan
def find_parent(parent, x) :
    if parent[x] != x :
        parent[x] = find_parent(parent, parent[x])
    return parent[x]

def union_parent(parent, a, b) :
    a = find_parent(parent, a)
    b = find_parent(parent, b)
    if a > b :
        parent[a] = b
    else :
        parent[b] = a


N, M = map(int, input().split())

edges = []
parent = [0] * (N + 1)
krus, result, last = [], 0, 0

for i in range(1, N + 1) :
    parent[i] = i

for _ in range(M) :
    A, B, C = map(int, input().split())
    edges.append((C, A, B))

edges.sort()

for edge in edges :
    cost, A, B = edge
    if find_parent(parent, A) != find_parent(parent, B) :
        union_parent(parent, A, B)
        result += cost
        last = cost

print(result - last)
```

<br>

#### 커리큘럼

``` curriculum
N = int(input())

time = [0] * N
prerequisites = [[] for _ in range(N)]

for i in range(N) :
    data = list(map(int, input().split()))
    time[i] = data[0]
    prerequisites[i] = data[1:-1]

subjecting, time_sum = [], 0
for i in range(N) :
    if not prerequisites[i] :
        subjecting.append([time[i], i])

while subjecting :
    min_time = int(1e9)
    for i in subjecting :
        if min_time > i[0] :
            min_time = i[0]
    time_sum += min_time

    for i in range(len(subjecting)) :
        subjecting[i][0] -= min_time
        if subjecting[i][0] == 0 :
            for j in range(N) :
                if (subjecting[i][1] + 1) in prerequisites[j] :
                    prerequisites[j].remove(subjecting[i][1] + 1)
                    if not prerequisites[j] :
                        subjecting.append([time[j], j])
            time[subjecting[i][1]] = time_sum
            subjecting.remove(subjecting[i])

print(time)
```

<br>

## Practice Solution

<br>

### Greedy Solution


#### Adventure's Guild

``` Adventure's Guild
N = int(input())

fear_rate = list(map(int, input().split()))
rate_array = [0 for _ in range(0, N)]
group_count = 0

for i in range(0, N) :
    rate_array[fear_rate[i]] += 1

for i in range(1, N) :
    group_count += rate_array[i] // i

print(group_count)
```

<br>

#### Multiple or Add

``` multiple or add
S = input()

result = int(S[0])

for i in range(1, len(S)) :
    num = int(S[i])
    if num <= 1 or result <= 1 :
        result += num
    else :
        result *= num

print(result)
```

<br>

#### Reverse String

``` reverse string
S = input()

origin_state = True
reverse_count = 0
current_number = S[0]

for i in range(1,len(S)) :
    if current_number != S[i] and origin_state:
        reverse_count += 1
        origin_state = False
    if current_number == S[i] :
        origin_state = True

print(reverse_count)
```

<br>

#### Don't make Money

``` don't make Money
N = int(input())
money_unit = list(map(int, input().split()))
money_unit.sort()

target = 1

for x in money_unit :
    if target < x :
        break
    target += x

print(target)
```

<br>

#### Choose the bowling ball

``` choose the bowling ball
N, M = map(int, input().split())
ball_weights = list(map(int, input().split()))

case_count = 0
weights = [0] * (M + 1)

for x in ball_weights :
    weights[x] += 1

for i in range(1, M + 1) :
    N = N - weights[i]
    case_count += N * weights[i]

print(case_count)
```

<br>

#### mukBang live

``` mukBang live
import heapq

def solution(food_times, k):
    food_count = len(food_times)
    time_now = 0
    q = []
    for i in range(0,food_count) :
        heapq.heappush(q, (food_times[i], i + 1))

    while True :
        time, num = heapq.heappop(q)

        if k > time_now + (time * (len(q) + 1)) :
            time_now += time * (len(q) + 1)
        else :
            time, num = q[(k - time_now - 1) % len(q)]
            return num


print("정전이후 먹어야 될 음식의 번호 : " + str(solution([3, 1, 2], 5)) + "번")
```

<br>

### Avatar Solution

<br>

#### lucky straight

``` lucky straight
N = input()
half_index = int(len(N) / 2)

left_sum = 0
right_sum = 0

for i in range(0, half_index) :
    left_sum += int(N[i])
    right_sum += int(N[i + half_index])

if left_sum == right_sum :
    print("LUCKY")
else :
    print("READY")
```

<br>

#### String Resort

``` string reSort
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
```

<br>

#### Compress String

``` compress string
def solution(s) :
    s_len = len(s)
    if s_len <= 1 :
        return s_len

    minLen = int(1e9)

    for length in range(1, s_len // 2 + 1):
        compressStr = ""
        sameCount = 1
        preStr = s[0:length]

        for i in range(length,s_len + 1,length) :
            if s_len >= i + length :
                if preStr == s[i: i + length]:
                    sameCount += 1
                else:
                    if sameCount != 1 :
                        compressStr += str(sameCount) + preStr
                    else :
                        compressStr += preStr

                    preStr = s[i: i + length]
                    sameCount = 1
            else :
                if sameCount != 1:
                    compressStr += str(sameCount) + preStr
                else :
                    compressStr += preStr

                compressStr += s[i : s_len]

        minLen = min(minLen, len(compressStr))

    return minLen


s = "aabbaccc"
print(solution(s))

s = "ababcdcdababcdcd"
print(solution(s))

s = "abcabcdede"
print(solution(s))

s = "abcabcabcabcdededededede"
print(solution(s))

s = "xababcdcdababcdcd"
print(solution(s))

s = "a"
print(solution(s))
```

<br>

#### Lock and Key

``` lock and key
def rotaition(key) :
    newKey = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(len(key)):
        for j in range(len(key)):
            if key[i][j] == 1:
                newKey[len(key) - j - 1][i] = 1
    return newKey

def check(lock) :
    lock_length = len(lock) // 3
    for i in range(lock_length, 2 * lock_length) :
        for j in range(lock_length, 2 * lock_length) :
            if lock[i][j] != 1 :
                return False

    return True

def solution(key, lock):
    key_size = len(key)
    lock_size = len(lock)
    new_lock = [[0] * (lock_size * 3) for _ in range(lock_size * 3)]

    for i in range(lock_size) :
        for j in range(lock_size):
            new_lock[lock_size + i][lock_size + j] = lock[i][j]

    for ro in range(4) :
        key = rotaition(key)
        for x in range(lock_size * 2) :
            for y in range(lock_size * 2) :
                for i in range(key_size) :
                    for j in range(key_size) :
                        new_lock[x + i][y + j] += key[i][j]

                if check(new_lock) :
                    return True

                for i in range(lock_size) : 
                    for j in range(lock_size) :
                        new_lock[x + i][y + j] -= key[i][j]

    return False

key = [[0,0,0], [1,0,0], [0,1,1]]
lock = [[1,1,1], [1,1,0], [1,0,1]]

print(solution(key, lock))
```

<br>

#### Snake

``` snake
N = int(input())
K = int(input())

board = [[0] * N for _ in range(N)]
for i in range(K) :
    row, column = map(int, input().split())
    board[row - 1][column - 1] = 1

L = int(input())

direction = []
for i in range(L) :
    X, C = map(str, input().split())
    direction.append((X,C))

x, y, time, direct = 0, 0, 0, 2
mx = [-1,0,1,0]
my = [0,-1,0,1]

snake_body = [(0,0)]

while True :
    time += 1
    state = False

    x, y = x + mx[direct], y + my[direct]

    if x < 0 or y < 0 or x >= N or y >= N :
        print("wall break")
        break

    for fat in snake_body :
        if fat[0] == x and fat[1] == y :
            print("snake_body break")
            state = True
    if state :
        break

    snake_body.append((x,y))
    if board[y][x] != 1 :
        snake_body.pop(0)

    if direction and int(direction[0][0]) == time :
        C = direction[0][1]
        direction.pop(0)
        if C == 'L' :
            direct = (direct + 4 - 1) % 4
        else :
            direct = (direct + 4 + 1) % 4

print(time)
```

<br>

#### install pillar and floor

```install pillar and floor
def possible(build) :
    for x, y, a in build :
        if a == 0 :  # 기둥
            if y == 0 or [x, y - 1, 0] in build or [x, y, 1] in build or [x - 1, y, 1] in build :
                continue
            return False
        else:  # 보
            if [x, y - 1, 0] in build or [x + 1, y - 1, 0] in build or ([x - 1, y, 1] in build and [x + 1, y, 1] in build) :
                continue
            return False
    return True

def solution(n, build_frame):
    build = []
    for frame in build_frame :
        x, y, a, b = frame
        if b == 0 :
            build.remove([x,y,a])
            if not possible(build) :
                build.append([x,y,a])
        else :
            build.append([x,y,a])
            if not possible(build) :
                build.remove([x,y,a])

    return sorted(build)

n = 5
build_frame = [[1,0,0,1],[1,1,1,1],[2,1,0,1],[2,2,1,1],[5,0,0,1],[5,1,0,1],[4,2,1,1],[3,2,1,1]]
# build_frame = [[0,0,0,1],[2,0,0,1],[4,0,0,1],[0,1,1,1],[1,1,1,1],[2,1,1,1],[3,1,1,1],[2,0,0,0],[1,1,1,0],[2,2,0,1]]
print(solution(n, build_frame))
```

<br>

#### Chicken Distance 

``` chicken distance
from itertools import combinations

N, M = map(int, input().split())
restaurant, house, min_sum = [], [], int(1e9)

for i in range(N) :
    city = list(map(int, input().split()))
    for j in range(N) :
        if city[j] == 1 :
            house.append((i,j))
        elif city[j] == 2 :
            restaurant.append((i, j))

combin = list(combinations(restaurant,M))

for restaurants in combin :
    sum_d = 0

    for hx, hy in house :
        min_d = int(1e9)
        for rx, ry in restaurants:
            min_d = min(min_d, abs(hx - rx) + abs(hy - ry))
        sum_d += min_d

    min_sum = min(sum_d,min_sum)

print(min_sum)
```

<br>

#### Exterior wall inspection

``` exterior wall inspection
from itertools import permutations

def solution(n, weak, dist):
    length = len(weak)
    for i in range(length) :
        weak.append(weak[i] + n)
    answer = len(dist) + 1

    for start in range(length) :
        for friends in list(permutations(dist, len(dist))) :
            count = 1
            position = weak[start] + friends[count - 1]
            for index in range(start, start + length) :
                if position < weak[index] :
                    count += 1
                    if count > len(dist) :
                        break
                    position = weak[index] + friends[count - 1]
            answer = min(answer, count)
    if answer > len(dist) :
        return -1
    return answer


n = 12
weak = [1,5,6,10]
dist = [1,2,3,4]
# weak = [1,3,4,9,10]
# dist = [3,5,7]
print(solution(n,weak,dist))
```

<br>

### Bfs / Dfs

<br>

#### Find City

``` find city
from collections import deque

N, M, K, X = map(int, input().split())

graph = [[] for _ in range(N + 1)]
visited = [False] * (N + 1)

for _ in range(M) :
    A, B = map(int, input().split())
    graph[A].append(B)

dist_to_node = []
queue = deque()
queue.append((X,0))
visited[X] = True

while queue :
    n, dist = queue.popleft()
    dist_to_node.append((n, dist))
    for i in graph[n] :
        if not visited[i] :
            queue.append((i, dist + 1))
            visited[i] = True

for i in range(N) :
    if dist_to_node[i][1] == K :
        visited[K] = False
        print(i + 1, end=' ')

if visited[K] :
    print(-1)
```

<br>

#### labortory

``` labortory
from itertools import combinations
import copy

def check_zero(maps) :
    zero_sum = 0
    for i in range(len(maps)) :
        for j in range(len(maps[0])) :
            if maps[i][j] == 0 :
                zero_sum += 1

    return zero_sum

def dfs(maps, row, colum) :
    if row - 1 >= 0 and maps[row - 1][colum] == 0:
        maps[row - 1][colum] = 2
        dfs(maps, row - 1, colum)
    if row + 1 < len(maps) and maps[row + 1][colum] == 0:
        maps[row + 1][colum] = 2
        dfs(maps, row + 1, colum)
    if colum - 1 >= 0 and maps[row][colum - 1] == 0:
        maps[row][colum - 1] = 2
        dfs(maps, row, colum - 1)
    if colum + 1 < len(maps[0]) and maps[row][colum + 1] == 0:
        maps[row][colum + 1] = 2
        dfs(maps, row, colum + 1)

    return maps


N, M = map(int, input().split())
maps, wall_list, virus_list = [], [], []
max_zero = 0

for i in range(N) :
    keep = list(map(int, input().split()))
    maps.append(keep)
    for j in range(M) :
        if keep[j] == 0 :
            wall_list.append((i,j))
        elif keep[j] == 2 :
            virus_list.append((i,j))

for walls in combinations(wall_list,3) :
    copy_map = copy.deepcopy(maps)
    copy_map[walls[0][0]][walls[0][1]] = 1
    copy_map[walls[1][0]][walls[1][1]] = 1
    copy_map[walls[2][0]][walls[2][1]] = 1

    for row, colum in virus_list:
        copy_map = dfs(copy_map, row, colum)

    max_zero = max(check_zero(copy_map), max_zero)

print(max_zero)
```

<br>

#### Economic Infection

``` economic infection
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
```

<br>

#### Bracket Traslation

``` bracket traslation
def is_balanced(p) :
    count = 0
    for i in range(len(p)) :
        if p[i] == '(' :
            count += 1
        else :
            count -= 1
        if count == 0 :
            return i

def is_perfected(p) :
    count = 0
    for i in range(len(p)) :
        if p[i] == '(' :
            count += 1
        else :
            count -= 1
        if count < 0 :
            return False
    return True

def solution(p):
    answer = ""
    if p == "" :
        return ""

    balance_index = is_balanced(p)
    u = p[:balance_index + 1]
    v = p[balance_index + 1:]

    if is_perfected(u) :
        answer = u + solution(v)
    else :
        answer = '('
        answer += solution(v)
        answer += ')'
        u = list(u[1:-1])
        for i in range(len(u)) :
            if u[i] == '(' :
                u[i] = ')'
            else :
                u[i] = '('
        answer += "".join(u)
    return answer

p = "(()())()"
p = ")("
p = "()))((()"
print(solution(p))
```

<br>

#### Insert Operation 

``` insert operation 
from itertools import permutations

def calculation(number_list, oper) :
    sum = number_list[0]
    for i in range(len(oper)) :
        if oper[i] == '+':
            sum += number_list[i + 1]
        elif oper[i] == '-':
            sum -= number_list[i + 1]
        elif oper[i] == '*':
            sum *= number_list[i + 1]
        else:
            if sum > 0:
                sum //= number_list[i + 1]
            else:
                sum = ((sum * -1) // number_list[i + 1]) * -1

    return sum


N = int(input())
number_list = list(map(int,input().split()))
oper_list = list(map(int, input().split()))
opers = ['+','-','*','/']
liner_oper_list = []
for i in range(len(oper_list)) :
    for j in range(oper_list[i]) :
        liner_oper_list.append(opers[i])

max_result = 0
min_result = int(1e9)

for oper in permutations(liner_oper_list,len(liner_oper_list)) :
    result = calculation(number_list, oper)
    if result > max_result :
        max_result = result
    if result < min_result :
        min_result = result

print(max_result)
print(min_result)
```

<br>

#### Avoid Monitor 

``` avoid monitor 
from itertools import combinations
import copy

N = int(input())
corridor, blank_list, student_list, teacher_list  = [], [], [], []
hang_state = True

for i in range(N) :
    row_corridor = list(map(str, input().split()))
    corridor.append(row_corridor)
    for j in range(N) :
        if row_corridor[j] == 'X' :
            blank_list.append((i,j))
        elif row_corridor[j] == 'S' :
            student_list.append((i,j))
        elif row_corridor[j] == 'T':
            teacher_list.append((i, j))


def start_monitor(corridor, row, colum):
    if row - 1 >= 0 and corridor[row - 1][colum] != 'T' :
        if corridor[row - 1][colum] == 'S':
            return False
        else :
            corridor[row - 1][colum] = 'T'
            if not start_monitor(corridor, row - 1, colum) :
                return False

    if row + 1 < N and corridor[row + 1][colum] != 'T' :
        if corridor[row + 1][colum] == 'S' :
            return False
        else :
            corridor[row + 1][colum] = 'T'
            if not start_monitor(corridor,row + 1, colum) :
                return False

    if colum - 1 >= 0 and corridor[row][colum - 1] != 'T' :
        if corridor[row][colum - 1] == 'S':
            return False
        else :
            corridor[row][colum - 1] = 'T'
            if not start_monitor(corridor, row, colum - 1) :
                return False
    if colum + 1 < N and corridor[row][colum + 1] != 'T' :
        if corridor[row][colum + 1] == 'S' :
            return False
        else :
            corridor[row][colum + 1] = 'T'
            if not start_monitor(corridor, row, colum + 1) :
                return False
    return True


def check_case() :
    for obstacle in combinations(blank_list, 3):
        corridor[obstacle[0][0]][obstacle[0][1]] = 'T'
        corridor[obstacle[1][0]][obstacle[1][1]] = 'T'
        corridor[obstacle[2][0]][obstacle[2][1]] = 'T'
        copy_corr = copy.deepcopy(corridor)

        for row, colum in teacher_list:
            hang_state = start_monitor(copy_corr, row, colum)
            if not hang_state:
                break

        if hang_state:
            return "Yes"
    return "No"

print(check_case())
```

<br>

#### Population Migration

``` population migration
N, L, R = map(int, input().split())

A, new_union, union, move_state, count = [], [], [], False, 0

for _ in range(N) :
    A.append(list(map(int, input().split())))


def move_country(s_row, s_colum):
    global union_people, union
    if s_row - 1 >= 0 and L <= abs(A[s_row][s_colum] - A[s_row - 1][s_colum]) <= R and not (s_row - 1, s_colum) in union :
        new_union.append((s_row - 1, s_colum))
        union.append((s_row - 1, s_colum))
        move_country(s_row - 1, s_colum)
    if s_colum - 1 >= 0 and L <= abs(A[s_row][s_colum] - A[s_row][s_colum - 1]) <= R and not (s_row, s_colum - 1) in union:
        new_union.append((s_row, s_colum - 1))
        union.append((s_row, s_colum - 1))
        move_country(s_row, s_colum - 1)
    if s_row + 1 < N and L <= abs(A[s_row][s_colum] - A[s_row + 1][s_colum]) <= R and not (s_row + 1, s_colum) in union:
        new_union.append((s_row + 1, s_colum))
        union.append((s_row + 1, s_colum))
        move_country(s_row + 1, s_colum)
    if s_colum + 1 < N and L <= abs(A[s_row][s_colum] - A[s_row][s_colum + 1]) <= R and not (s_row, s_colum + 1) in union:
        new_union.append((s_row, s_colum + 1))
        union.append((s_row, s_colum + 1))
        move_country(s_row, s_colum + 1)


while True:
    union, move_state = [], False
    for s_row in range(N) :
        for s_colum in range(N):
            if not (s_row, s_colum) in union:
                new_union, sum_people = [], 0

                union.append((s_row, s_colum))
                new_union.append((s_row, s_colum))

                move_country(s_row, s_colum)

                if len(new_union) >= 2:
                    move_state = True
                    for row, colum in new_union:
                        sum_people += A[row][colum]

                    distribution = sum_people // (len(new_union))

                    for row, colum in new_union:
                        A[row][colum] = distribution
                else:
                    union.remove((s_row, s_colum))

    if not move_state:
        break
    else :
        count += 1

print(count)
```

<br>

#### Move Blocks

``` move blocks
from collections import deque

def get_next_pos(pos, board) :
    next_pos = []
    pos = list(pos)
    pos1_x, pos1_y, pos2_x, pos2_y = pos[0][0], pos[0][1], pos[1][0], pos[1][1]

    dx = [-1,1,0,0]
    dy = [0,0,-1,1]

    for i in range(4) :
        pos1_next_x, pos1_next_y, pos2_next_x, pos2_next_y = pos1_x + dx[i], pos1_y + dy[i], pos2_x + dx[i], pos2_y + dy[i]

        if board[pos1_next_x][pos1_next_y] == 0 and board[pos2_next_x][pos2_next_y] == 0 :
            next_pos.append({(pos1_next_x, pos1_next_y), (pos2_next_x, pos2_next_y)})

    if pos1_x == pos2_x :
        for i in [-1,1] :
            if board[pos1_x + i][pos1_y] == 0 and board[pos2_x + i][pos2_y] == 0 :
                next_pos.append({(pos1_x, pos1_y), (pos1_x + i, pos1_y)})
                next_pos.append({(pos2_x, pos2_y), (pos2_x + i, pos2_y)})
    elif pos1_y == pos2_y :
        for i in [-1,1] :
            if board[pos1_x][pos1_y + i] == 0 and board[pos2_x][pos2_y + i] == 0 :
                next_pos.append({(pos1_x, pos1_y), (pos1_x, pos1_y + i)})
                next_pos.append({(pos2_x, pos2_y), (pos2_x, pos2_y + i)})

    return next_pos

def solution(board):
    n = len(board)

    new_board = [[1] * (n + 2) for _ in range(n + 2)]
    for i in range(n) :
        for j in range(n) :
            new_board[i + 1][j + 1] = board[i][j]

    q = deque()
    visited = []
    pos = {(1,1),(1,2)}
    q.append((pos, 0))
    visited.append(pos)

    while q :
        pos, cost = q.popleft()

        if (n,n) in pos :
            return cost

        for next_pos in get_next_pos(pos, new_board) :
            if next_pos not in visited :
                q.append((next_pos, cost + 1))
                visited.append(next_pos)
    return 0


board = [[0, 0, 0, 1, 1],[0, 0, 0, 1, 0],[0, 1, 0, 1, 1],[1, 1, 0, 0, 1],[0, 0, 0, 0, 0]]
print(solution(board))
```

<br>

### Sort

<br>

#### Subject Sort

``` subject sort
N = int(input())
student_info = []

for _ in range(N) :
    student_info.append(input().split())

student_info.sort(key=lambda x : (-int(x[1]) , int(x[2]), -int(x[3]), x[0]))

for student in student_info :
    print(student[0])
```

<br>

#### Antenna

``` antenna
N = int(input())
house_list = list(map(int, input().split()))
house_list.sort()

print(house_list[(N-1) // 2])
```

<br>

#### Fail Rate

``` failrate
def solution(N, stages):
    failrate_stage = []
    stage_size = len(stages)

    for i in range(1, N + 1) :
        count = stages.count(i)

        if stage_size == 0 :
            failrate_stage.append((0,i))
        else :
            failrate_stage.append((count / stage_size, i))
            stage_size -= count

    failrate_stage.sort(key=lambda x : (-x[0], x[1]))

    return [i[1] for i in failrate_stage]

N = 5
stages = [2,1,2,6,2,4,3,3]
print(solution(N, stages))

N = 4
stages = [4,4,4,4,4]
print(solution(N, stages))
```

<br>

#### Card Sort

``` card sort
import heapq

N = int(input())
card_bundle = []

for _ in range(N) :
    heapq.heappush(card_bundle, int(input()))

sum_compare = heapq.heappop(card_bundle) * (N - 1)

for i in range(1,N) :
    sum_compare += heapq.heappop(card_bundle) * (N - i)

print(sum_compare)
```

<br>

#### Find number in sort arr

```find number in sort arr
from bisect import bisect_left, bisect_right

def count_by_range(arr, left_value, right_value) :
    right_index = bisect_right(arr, right_value)
    left_index = bisect_left(arr, left_value)
    return right_index - left_index


N, x = map(int, input().split())
arr = list(map(int, input().split()))

count = count_by_range(arr, x, x)

if count == 0 :
    print(-1)
else :
    print(count)
```

<br>

#### Find fix pos

``` find fix pos
def binary_search(arr, start, end) :
    mid = (start + end) // 2

    if start > end :
        return None

    if arr[mid] > mid :
        return binary_search(arr, start, mid - 1)
    elif arr[mid] < mid :
        return binary_search(arr, mid + 1, end)
    else :
        return mid

N = int(input())
arr = list(map(int, input().split()))

print(binary_search(arr, 0, N - 1))
```

<br>

#### Set Router

``` set router
N, C = map(int, input().split())
house_loc, result = [], 0
for _ in range(N) :
    house_loc.append(int(input()))

house_loc.sort()
start = 1
end = house_loc[-1] - house_loc[0]

while(start <= end) :
    mid = (start + end) // 2
    pre, setRouter = house_loc[0], 1

    for i in range(1, N) :
        if house_loc[i] - pre >= mid :
            pre = house_loc[i]
            setRouter += 1

    if setRouter >= C :
        start = mid + 1
        result = mid
    else :
        end = mid - 1

print(result)
```

<br>

#### Lyrics Search

``` lyrics search
from bisect import bisect_left, bisect_right

def count_by_range(a, left_value, right_value) :
    right_index = bisect_right(a, right_value)
    left_index = bisect_left(a, left_value)
    return right_index - left_index


array = [[] for _ in range(10001)]
reversed_array = [[] for _ in range(10001)]


def solution(words, queries):
    answer = []

    for word in words :
        array[len(word)].append(word)
        reversed_array[len(word)].append(word[::-1])

    for i in range(10001) :
        array[i].sort()
        reversed_array[i].sort()

    for query in queries :
        if query[0] == '?' :
            res = count_by_range(reversed_array[len(query)], query[::-1].replace('?','a'), query[::-1].replace('?','z'))
        else :
            res = count_by_range(array[len(query)], query.replace('?','a'), query.replace('?','z'))

        answer.append(res)

    return answer


words = ["frodo", "front", "frost", "frozen", "frame", "kakao"]
queries = ["fro??", "????o", "fr???", "fro???", "pro?"]
print(solution(words, queries))
```

<br>

#### Gold Mine

``` gold mine
T = int(input())
max_result = []
for cycle in range(T) :
    n, m = map(int, input().split())
    lists = list(map(int, input().split()))
    maps, max_maps = [], [[0] * n for _ in range(m)]

    for i in range(n):
        maps.append(lists[m * i: m * (i + 1)])
        max_maps[0][i] = lists[m * i]

    for i in range(1, m):
        for j in range(n):
            if j - 1 >= 0 and j + 1 < n:
                max_maps[i][j] = maps[j][i] + max(max_maps[i - 1][j - 1: j + 2])
            elif j - 1 < 0:
                max_maps[i][j] = maps[j][i] + max(max_maps[i - 1][j: j + 2])
            else:
                max_maps[i][j] = maps[j][i] + max(max_maps[i - 1][j - 1: j + 1])

    max_result.append(max(max_maps[m - 1]))

for result in max_result :
    print(result)
```

<br>

#### Integer Triangle

``` integer triangle
n = int(input())

tri = []
max_tri = [[0] * i for i in range(1,n + 1)]

for _ in range(n) :
    tri.append(list(map(int, input().split())))

max_tri[0][0] = tri[0][0]

for i in range(1, n) :
    for j in range(i + 1) :
        if j - 1 >= 0 and j < i :
            max_tri[i][j] = tri[i][j] + max(max_tri[i - 1][j], max_tri[i - 1][j - 1])
        elif j - 1 < 0 :
            max_tri[i][j] = tri[i][j] + max_tri[i - 1][j]
        else :
            max_tri[i][j] = tri[i][j] + max_tri[i - 1][j - 1]

print(max(max_tri[n - 1]))
```

<br>

#### Resign

``` resign
N = int(input())
consulting = []
max_consulting = [0] * (N + 1)

for _ in range(N) :
    consulting.append(list(map(int, input().split())))

for i in range(N) :
    target_index = consulting[i][0] + i
    if target_index < N + 1 :
        for j in range(target_index, N + 1) :
            max_consulting[target_index] = max(consulting[i][1] + max_consulting[i], max_consulting[target_index])
            if max_consulting[j] < max_consulting[target_index] :
                max_consulting[j] = max_consulting[target_index]


print(max(max_consulting))
```

<br>

#### Collocate Soldier

``` collocate soldier
N = int(input())
soldier_list = list(map(int,input().split()))

dp = [1] * N

for i in range(N) :
    for j in range(i) :
        if soldier_list[i] < soldier_list[j] :
            dp[i] = max(dp[i], dp[j] + 1)

print(N - max(dp))
```

<br>

#### Ugly Number

``` ugly number
n = int(input())
measures = [2,3,5]
ugly_number = [1]
index = 0

while len(ugly_number) <= n :
    for measure in measures :
        if not ugly_number[index] * measure in ugly_number :
            ugly_number.append(ugly_number[index] * measure)
    index += 1

ugly_number.sort()
print(ugly_number[n - 1])
```

<br>

#### Edit Distance

``` edit distance
A = input()
B = input()

A_len, B_len, state = len(A), len(B), False
dp = [[0] * B_len for _ in range(A_len)]

for i in range(A_len) :
    dp[i][0] = i + 1
    if state or B[0] == A[i] :
        state = True
        dp[i][0] -= 1

state = False
for i in range(B_len) :
    dp[0][i] = i + 1
    if state or A[0] == B[i] :
        state = True
        dp[0][i] -= 1


for i in range(1, A_len) :
    for j in range(1, B_len) :
        if A[i] == B[j] :
            dp[i][j] = dp[i - 1][j - 1]
        else :
            dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

print(dp[A_len - 1][B_len - 1])
```

<br>

#### Floyd

``` floyd
n = int(input())
m = int(input())

graph = [[int(1e9)] * (n + 1) for _ in range(n + 1)]

for i in range(1, n + 1) :
    graph[i][i] = 0

for _ in range(m) :
    a, b, c = map(int, input().split())
    if c < graph[a][b] :
        graph[a][b] = c

for k in range(1, n + 1) :
    for i in range(1, n + 1) :
        for j in range(1, n + 1) :
            graph[i][j] = min(graph[i][j], graph[i][k] + graph[k][j])

for i in range(1, n + 1) :
    for j in range(1, n + 1) :
        print(graph[i][j] , end=" ")
    print()
```

<br>

#### Exact Ranking

``` exact ranking
N, M = map(int, input().split())
result_sum = 0
result = [0] * (N + 1)

graph = [[int(1e9)] * (N + 1) for _ in range(N + 1)]

for i in range(N + 1) :
    graph[i][i] = 0

for _ in range(M) :
    A, B = map(int, input().split())
    graph[A][B] = 1

for k in range(1, N + 1) :
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if graph[i][k] == 1 and graph[k][j] == 1 :
                graph[i][j] = 1


for i in range(1, N + 1) :
    for j in range(1, N + 1) :
        if graph[i][j] == 1 :
            result[i] += 1
            result[j] += 1

for i in result :
    if i == 5 :
        result_sum += 1

print(result_sum)
```

<br>

#### Mars Exploration

``` mars exploration
import heapq

T = int(input())
dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]
result = []

for _ in range(T) :
    N = int(input())

    graph = []
    distance = [[int(1e9)] * N for _ in range(N)]

    for _ in range(N) :
        graph.append(list(map(int, input().split())))

    q = []
    distance[0][0] = graph[0][0]
    heapq.heappush(q, (graph[0][0], 0, 0))

    while q:
        dist, nx, ny = heapq.heappop(q)

        if dist > distance[nx][ny]:
            continue

        for i in range(4):
            cx = nx + dx[i]
            cy = ny + dy[i]
            if cx < 0 or cx >= N or cy < 0 or cy >= N:
                continue

            cost = dist + graph[cx][cy]
            if distance[cx][cy] > cost:
                distance[cx][cy] = cost
                heapq.heappush(q, (cost, cx, cy))

    result.append(distance[N - 1][N - 1])

print(result)
```

<br>

#### Hide and Seek

``` hide and seek
import heapq

N,  M = map(int, input().split())

graph = [[] for _ in range(N + 1)]
for _ in range(M) :
    A, B = map(int, input().split())
    graph[A].append((B, 1))
    graph[B].append((A, 1))

distances = [int(1e9)] * (N + 1)

def dijkstra(start) :
    q = []
    heapq.heappush(q, (0, start))
    distances[start] = 0

    while q :
        dist, now = heapq.heappop(q)

        if dist > distances[now] :
            continue

        for i in graph[now] :
            cost = dist + i[1]
            if distances[i[0]] > cost :
                distances[i[0]] = cost
                heapq.heappush(q, (cost, i[0]))

    return distances

distance_case = dijkstra(1)
distance_case.remove(int(1e9))
max_sel, max_index, max_same = distance_case[0], 0, 0

for i in range(1, N) :
    if distance_case[i] > max_sel :
        max_sel = distance_case[i]
        max_index = i

for i in range(N) :
    if distance_case[i] == max_sel :
        max_same += 1

print(str(max_index + 1) + " " + str(max_sel) + " " + str(max_same))
```

<br>

#### Trip plan

``` trip plan
def find_parent(parent, x) :
    if parent[x] != x :
        parent[x] = find_parent(parent, parent[x])
    return parent[x]


def union_parent(parent, a, b) :
    a = find_parent(parent, a)
    b = find_parent(parent, b)
    if a > b :
        parent[a] = b
    else :
        parent[b] = a


N, M = map(int, input().split())

parent, data, state = [0] * N, [], True

for i in range(N) :
    parent[i] = i

for i in range(N) :
    info = list(map(int, input().split()))
    for j in range(N) :
        if info[j] == 1 and not [j,i] in data :
            data.append([i,j])
            union_parent(parent, i, j)

plans = list(map(int, input().split()))

for i in range(N - 1) :
    if find_parent(parent, i) != find_parent(parent, i + 1) :
        state = False

if state :
    print("YES")
else :
    print("NO")
```

<br>

#### Dark road

``` dark road
def find_parent(parent, x) :
    if parent[x] != x:
        parent[x] = find_parent(parent, parent[x])
    return parent[x]

def union_parent(parent, a, b) :
    a = find_parent(parent, a)
    b = find_parent(parent, b)
    if a > b :
        parent[a] = b
    else :
        parent[b] = a


N, M = map(int, input().split())

parent, edges, cost_sum, origin_cost = [0] * N, [], 0, 0

for i in range(N) :
    parent[i] = i

for _ in range(M) :
    X, Y, Z = map(int, input().split())
    origin_cost += Z
    edges.append((Z, X, Y))

edges.sort()

for edge in edges :
    if find_parent(parent, edge[1]) != find_parent(parent, edge[2]) :
        cost_sum += edge[0]
        union_parent(parent, edge[1], edge[2])

print(origin_cost - cost_sum)
```

<br>

#### Gate

``` gate
def find_parent(parent, x) :
    if parent[x] != x :
        parent[x] = find_parent(parent, parent[x])
    return parent[x]


def union_parent(parent, a, b) :
    a = find_parent(parent, a)
    b = find_parent(parent, b)
    if a > b :
        parent[a] = b
    else :
        parent[b] = a


G = int(input())
P = int(input())

parent = [0] * (G + 1)
result = 0

for i in range(1, G + 1) :
    parent[i] = i

for _ in range(P) :
    data = find_parent(parent, int(input()))
    if data == 0 :
        break

    union_parent(parent, data, data - 1)
    result += 1

print(result)
```

<br><br>

## Programmers

#### Get report results

``` get report results
def solution(id_list, report, k):
    id_size = len(id_list)
    reported = {id: [] for id in id_list}
    mail_list = {id: 0 for id in id_list}

    for content in set(report):
        repo, repoed = content.split()
        reported[repoed].append(repo)

    for key, value in reported.items():
        if len(value) >= k:
            for i in value:
                mail_list[i] += 1

    return mail_list.values()

id_list = ["muzi", "frodo", "apeach", "neo"]
report = ["muzi frodo","apeach frodo","frodo neo","muzi neo","apeach muzi"]
k = 2

# id_list = ["con", "ryan"]
# report = ["ryan con", "ryan con", "ryan con", "ryan con"]
# k = 3
print(solution(id_list, report, k))
```

배운점 :
- 중복제거 : set()
- 파이썬 딕셔너리 사용법 : {}
  - {n : ? for n in ~} 


<br>

#### Recommand New Id 

``` recommand new id
def solution(new_id):
    new_id = new_id.lower()
    imposs_char = "~!@#$%^&*()=+[{]}:?,<>/"

    for char in imposs_char:
        if char in new_id:
            new_id = new_id.replace(char, '')

    i = 0
    while i < len(new_id) :
        if new_id[i:i+2] == ".." :
            new_id = new_id[:i+1] + new_id[i+2:]
            i -= 1
        i +=1

    if new_id[0] == '.' :
        new_id = new_id[1:]

    new_id_size = len(new_id)
    if new_id and new_id[new_id_size - 1] == '.' :
        new_id = new_id[:new_id_size - 1]

    if not new_id:
        new_id = "a"

    if len(new_id) >= 16:
        new_id = new_id[:15]

    new_id_size = len(new_id)
    if new_id[new_id_size - 1] == '.':
        new_id = new_id[:new_id_size - 1]

    while len(new_id) <= 2:
        new_id += new_id[len(new_id) - 1]
    
    print(new_id)
    return new_id


solution("...!@BaT#*..y.abcdefghijklm")
solution("z-+.^.")
solution("=.=")
solution("123_.def")
solution("abcdefghijklmn.p")
```

<br>

#### Draw Doll

``` draw doll
def solution(board, moves):
    result, basket = 0, []
    for move in moves:
        for weight in board:
            if weight[move - 1] != 0:
                if basket and basket[len(basket) - 1] == weight[move - 1]:
                    basket.pop()
                    result += 2
                else :
                    basket.append(weight[move - 1])
                weight[move - 1] = 0
                break

    return result

board = [[0,0,0,0,0],[0,0,1,0,3],[0,2,5,0,1],[4,2,4,4,2],[3,5,1,3,1]]
moves = [1,5,3,5,1,2,1,4]
print(solution(board, moves))
```

<br>

#### K' th Number

``` K' th number
def solution(array, commands):
    answer = []
    for command in commands:
        cur_arr = array[command[0] - 1: command[1]]
        cur_arr.sort()
        answer.append(cur_arr[command[2] - 1])

    return answer

array = [1, 5, 2, 6, 3, 7, 4]
commands = [[2, 5, 3], [4, 4, 1], [1, 7, 3]]
print(solution(array, commands))
```

<br>

#### Gym Suit

``` gym suit
def solution(n, lost, reserve):
    set_reserve = set(reserve) - set(lost)
    set_lost = set(lost) - set(reserve)
    for number in set_lost:
        if number - 1 in set_reserve:
            set_reserve.remove(number - 1)
        elif number + 1 in set_reserve:
            set_reserve.remove(number + 1)
        else :
            n-=1

    return n

n = 5
lost = [2, 4]
reserve = [1, 3, 5]

n = 5
lost = [2, 4]
reserve = [3]

n = 3
lost = [3]
reserve = [1]

solution(n, lost, reserve)
```

<br>

#### Click Keypad

``` click keypad
def distance(x, y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])


def solution(numbers, hand):
    cur_left, cur_right = [3, 0], [3, 2]
    answer = ""
    for number in numbers:
        if number == 0:
            number = 10
        else:
            number -= 1
        row, colum = number // 3, number % 3

        if colum == 0:
            cur_left = [row, colum]
            answer += 'L'
        elif colum == 2:
            cur_right = [row, colum]
            answer += 'R'
        else:
            if distance((row, colum), cur_left) < distance((row, colum), cur_right):
                cur_left = [row, colum]
                answer += 'L'
            elif distance((row, colum), cur_left) > distance((row, colum), cur_right):
                cur_right = [row, colum]
                answer += 'R'
            else:
                if hand == "left":
                    cur_left = [row, colum]
                    answer += 'L'
                else:
                    cur_right = [row, colum]
                    answer += 'R'

    return answer

numbers = [1, 3, 4, 5, 8, 2, 1, 4, 5, 9, 5]
hand = "right"
print(solution(numbers, hand))

numbers = [7, 0, 8, 2, 8, 3, 1, 5, 7, 6, 2]
hand = "left"
print(solution(numbers, hand))

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
hand = "right"
print(solution(numbers, hand))
```

<br>

#### Open Chatting

``` open chatting
def solution(record):
    user = {}
    answers = []
    result = []
    for i in record:
        data = i.split()
        if data[0] == "Change":
            user[data[1]] = data[2]
        else:
            if data[0] == "Enter" :
                user[data[1]] = data[2]
            answers.append([data[0], data[1]])

    for answer in answers:
        if answer[0] == "Enter":
            result.append(user[answer[1]] + "님이 들어왔습니다.")
        else:
            result.append(user[answer[1]] + "님이 나갔습니다.")

    return result

arr = ["Enter uid1234 Muzi", "Enter uid4567 Prodo","Leave uid1234","Enter uid1234 Prodo","Change uid4567 Ryan"]
print(solution(arr))
```

<br>

#### Function Development

``` function development
from collections import deque

def solution(progresses, speeds):
    progresses, speeds = deque(progresses), deque(speeds)
    answer = []

    while progresses:
        index = 0
        for i in range(0, len(progresses)):
            progresses[i] = progresses[i] + speeds[i]

        while progresses and progresses[0] >= 100:
            progresses.popleft()
            speeds.popleft()
            index += 1

        if index != 0:
            answer.append(index)

    return answer

progresses = [93, 30, 55]
speeds = [1, 30, 5]
print(solution(progresses,speeds))

progresses = [95, 90, 99, 99, 80, 99]
speeds = [1, 1, 1, 1, 1, 1]
print(solution(progresses,speeds))
```

<br>

#### Target Number

``` target number
from collections import deque

def solution(numbers, target):
    total = deque([0])
    numbers = deque(numbers)
    answer = 0

    while numbers:
        number = numbers.popleft()
        for _ in range(len(total)):
            sel = total.popleft()
            total.append(sel - number)
            total.append(sel + number)

    for i in total:
        if i == target:
            answer += 1

    return answer

numbers   = [1, 1, 1, 1, 1]
target = 3
print(solution(numbers,target))
```

<br>

#### Menu Renewal

``` menu renewal
from itertools import combinations
from collections import Counter

def solution(orders, course):
    answer = []
    for course_case in course :
        menu_can, max_orders = [], 0
        for order in orders :
            for combin in combinations(order, course_case) :
                menu_can.append("".join(sorted(combin)))

        can_list = Counter(menu_can).most_common()
        if not can_list :
            continue
        max_orders = can_list[0][1]
        if max_orders < 2 :
            continue
        for can in  can_list:
            if can[1] != max_orders :
                break
            answer.append(can[0])

    return sorted(answer)

orders = ["ABCFG", "AC", "CDE", "ACDE", "BCFG", "ACDEH"]
course = [2,3,4]
print(solution(orders, course))

orders = ["ABCDE", "AB", "CD", "ADE", "XYZ", "XYZ", "ACD"]
course = [2,3,5]
print(solution(orders, course))

orders = ["XYZ", "XWY", "WXA"]
course = [2,3,4]
print(solution(orders, course))
```

배운점 :
- Counter 라이브러리

<br>

#### Printer

``` printer
from collections import deque

def check_priorities(priorities):
    first = priorities[0]
    for i in range(1, len(priorities)):
        if priorities[i] > first:
            return False
    return True


def solution(priorities, location):
    priorities = deque(priorities)
    pop_index = 0

    while True:
        while not check_priorities(priorities):
            priorities.rotate(-1)
            location -= 1
            if location < 0:
                location = len(priorities) - 1

        if location <= 0 and check_priorities(priorities) :
            break
        priorities.popleft()
        location -= 1
        pop_index += 1
        if location < 0:
            location = len(priorities) - 1

    return pop_index + 1

priorities = [2, 1, 3, 2]
location = 2
print(solution(priorities,location))

priorities = [1, 1, 9, 1, 1, 1]
location = 0
print(solution(priorities,location))
```

<br>

#### Big Number

``` big number
def solution(numbers):
    numbers = list(map(str, numbers))
    numbers.sort(key=lambda x : x * 3, reverse=True)
    print(numbers)

    return str(int("".join(numbers)))


numbers = [6, 10, 2]
print(solution(numbers))

numbers = [3, 30, 34, 5, 9]
print(solution(numbers))
```

배운점
 - 정렬방식

<br>

#### Find PrimeNumber

``` find primenumber
from itertools import permutations

def is_prime(number):
    if number == 2:
        return True
    elif number == 1 or number == 0 :
        return False
    for i in range(2, number):
        if number % i == 0:
            return False
    return True

def solution(numbers):
    answer = 0
    prime_list = []
    for i in range(1, len(numbers) + 1):
        for data in permutations(numbers, i):
            prime_list.append(int("".join(data)))

    prime_list = list(set(prime_list))
    for prime in prime_list:
        if is_prime(prime):
            answer += 1

    return answer

numbers = "17"
print(solution(numbers))

numbers = "011"
print(solution(numbers))
```


<br>

#### Farthest Node

``` farthest node
from collections import deque

def solution(n, edge) :
    graph = [[] for _ in range(n + 1)]
    visited = [False] * (n + 1)

    for edge_case in edge :
        graph[edge_case[0]].append(edge_case[1])
        graph[edge_case[1]].append(edge_case[0])

    queue = deque()
    queue.append([1, 0])
    answer, max_dist = 0, 0

    visited[1] = True
    while queue:
        v, dist = queue.popleft()

        if dist > max_dist:
            max_dist = dist
            answer = 1
        elif dist == max_dist:
            answer += 1

        for i in graph[v]:
            if not visited[i]:
                queue.append([i, dist + 1])
                visited[i] = True

    return answer

n = 6
vertex = [[3, 6], [4, 3], [3, 2], [1, 3], [1, 2], [2, 4], [5, 2]]
print(solution(n, vertex))
```

<br>

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

#### Immigration

``` immigration
def solution(n, times):
    start, end = 1, max(times) * n
    last = 0

    while start <= end:
        mid, result = (start + end) // 2, 0

        for time in times:
            result += mid // time

        if n > result:
            start = mid + 1
        else:
            last = mid
            end = mid - 1

    return last

n = 6
times = [7, 10]
print(solution(n, times))
```

<br>

#### Check the Distance 

``` check the distance
from collections import deque


def check_distance(arr, colum, row):
    queue = deque()
    queue.append([colum, row, 0])
    visited = [[False] * 5 for _ in range(5)]
    while queue:
        col, ro, dist = queue.popleft()
        visited[col][ro] = True

        if col - 1 >= 0 and not visited[col - 1][ro] and arr[col - 1][ro] != 'X':
            if arr[col - 1][ro] == 'P':
                return False
            if dist < 1 :
                queue.append([col - 1, ro, dist + 1])
        if ro - 1 >= 0 and not visited[col][ro - 1] and arr[col][ro - 1] != 'X':
            if arr[col][ro - 1] == 'P':
                return False
            if dist < 1 :
                queue.append([col, ro - 1, dist + 1])
        if col + 1 < 5 and not visited[col + 1][ro] and arr[col + 1][ro] != 'X':
            if arr[col + 1][ro] == 'P':
                return False
            if dist < 1:
                queue.append([col + 1, ro, dist + 1])
        if ro + 1 < 5 and not visited[col][ro + 1] and arr[col][ro + 1] != 'X':
            if arr[col][ro + 1] == 'P':
                return False
            if dist < 1:
                queue.append([col, ro + 1, dist + 1])

    return True


def solution(places):
    answer = []
    for place in places:
        state = True
        for colum in range(5):
            for row in range(5):
                if place[colum][row] == 'P':
                    state = check_distance(place, colum, row)
                if not state :
                    break
            if not state :
                break
        answer.append(int(state))
    return answer


places = [["POOOP", "OXXOX", "OPXPX", "OOXOX", "POXXP"], ["POOPX", "OXPXP", "PXXXO", "OXXXO", "OOOPP"], ["PXOPX", "OXOXP", "OXPOX", "OXXOP", "PXPOX"], ["OOOXX", "XOOOX", "OOOXX", "OXOOX", "OOOOO"], ["PXPXP", "XPXPX", "PXPXP", "XPXPX", "PXPXP"]]
print(solution(places))
```

<br>

#### Network

``` network
from collections import deque

def solution(n, computers):
    answer = 0

    for i in range(n):
        if computers[i][i] != 0:
            computers[i][i] = 0

            queue = deque([i])

            while queue:
                node = queue.popleft()
                computers[node][node]

                for i in range(n):
                    if computers[node][i] == 1:
                        queue.append(i)
                        computers[node][i] = 0
                        computers[i][node] = 0

            answer += 1

    return answer


n=3
computers = [[1, 1, 0], [1, 1, 0], [0, 0, 1]]
print(solution(n, computers))

n=3
computers = [[1, 1, 0], [1, 1, 1], [0, 1, 1]]
print(solution(n, computers))
```

<br>

#### Joy Stick

```joy stick
def solution(name):
    answer, n1, n2, name_size = 0, 0, 0, len(name)

    for i in range(name_size - 1, 0, -1) :
        if name[i] == 'A' :
            n1 += 1
        else :
            break

    for i in range(1, name_size, 1) :
        if name[i] == 'A' :
            n2 += 1
        else :
            break

    move = name_size - min(n1, n2) - 1 # 최대 이동거리

    for i in range(name_size) :
        answer += min(ord(name[i]) - 65, 91 - ord(name[i]))

        index = i
        while index < name_size and name[index] == 'A' :
            index += 1

        move = min(move, i * 2 + name_size - index, (i + 2 * (name_size - index)))

    answer += move
    return answer

name = "JEROEN"
print(solution(name))

name = "JAN"
print(solution(name))
```

<br>

#### Camouflage

``` camouflage
def solution(clothes):
    answer = 1
    cloth_combin = {}

    for cloth in clothes :
        if cloth[1] in cloth_combin.keys() :
            cloth_combin[cloth[1]] += 1
        else :
            cloth_combin[cloth[1]] = 1

    for num in cloth_combin.values() :
        answer *= (num + 1)

    return answer - 1

clothes = [["yellowhat", "headgear"], ["bluesunglasses", "eyewear"], ["green_turban", "headgear"]]
print(solution(clothes))

clothes = [["crowmask", "face"], ["bluesunglasses", "face"], ["smoky_makeup", "face"]]
print(solution(clothes))
```

<br>

#### tuple

```
def solution(s):
    answer = []
    s = (s[2:-2]).split("},{")
    s = sorted(s, key=lambda x : len(x))

    for i in s :
        k = i.split(',')
        for j in k :
            if int(j) not in answer :
                answer.append(int(j))
                break
    return answer

s = "{{2},{2,1,3},{2,1},{2,1,3,4}}"
print(solution(s))
```

<br>

#### Disk Controller

``` disk controller
import heapq

def solution(jobs):
    answer, cur_time, jobs_size = 0, 0, len(jobs)
    heap = []
    jobs.sort()

    while heap or jobs :
        while jobs and cur_time >= jobs[0][0]:
            process = jobs.pop(0)
            heapq.heappush(heap, [process[1], process[0]])

        if heap :
            process = heapq.heappop(heap)
            cur_time += process[0]
            answer += (cur_time - process[1])
        else :
            cur_time = jobs[0][0]

    return int(answer / jobs_size)

jobs = [[0, 3], [0, 2], [1, 9], [2, 6]]
print(solution(jobs))
```

<br>

#### Truck crossing the Bridge

``` truck crossing the bridge
from collections import deque

def solution(bridge_length, weight, truck_weights):
    answer, truck_sum = 0, 0
    bridge = deque([])

    while truck_weights or bridge :

        for truck in bridge:
            truck[1] += 1

        if bridge and bridge[0][1] >= bridge_length :
            escape_truck = bridge.popleft()
            truck_sum -= escape_truck[0]

        if truck_weights and bridge_length > len(bridge) and weight >= truck_sum + truck_weights[0]:
            in_bridge = truck_weights.pop(0)
            bridge.append([in_bridge, 0])
            truck_sum += in_bridge

        answer += 1

    return answer


bridge_length = 2
weight = 10
truck_weights = [7,4,5,6]
print(solution(bridge_length, weight, truck_weights))


bridge_length = 100
weight = 100
truck_weights = [10]
print(solution(bridge_length, weight, truck_weights))


bridge_length = 100
weight = 100
truck_weights = [10,10,10,10,10,10,10,10,10,10]
print(solution(bridge_length, weight, truck_weights))

bridge_length = 1
weight = 10
truck_weights = [1,1,1,1,1,1,1,1,1,1,1,1,1]
print(solution(bridge_length, weight, truck_weights))
``` 

<br>

#### Carpet

``` carpet
def solution(brown, yellow):
    answer = []

    w_h = brown // 2

    for i in range(1, w_h - 1) :
        weight = w_h - i
        height = w_h - weight + 2

        if yellow == (weight - 2) * (height - 2) :
            answer.append(weight)
            answer.append(height)
            break

    return answer

brown = 10
yellow = 2
print(solution(brown, yellow))
```

<br>

#### H-Index

``` H-Index
def solution(citations):
    answer, i, j = 0, 0, 0
    citations_size = len(citations)
    citations.sort()

    for i in range(1, citations_size + 1):
        for j in range(citations_size):
            if citations[j] >= i:
                break
            elif citations_size - 1 == j :
                j = citations_size
        if citations_size - j >= i:
            answer = i
        else:
            break

    return answer

citations = [5, 5, 5, 5, 5]
print(solution(citations))


citations = [0, 0, 0, 0, 0]
print(solution(citations))
```

<br>

#### Word Conversion

``` word conversion
from collections import deque


def check_go(word_1, word_2):
    index = 0
    for i in range(len(word_1)):
        if word_1[i] != word_2[i]:
            index += 1

    return 1 >= index


def solution(begin, target, words):
    answer, word_len, = 0, len(words)

    visited = [False] * word_len
    queue = deque([])

    for i in range(word_len):
        if check_go(begin, words[i]):
            queue.append([words[i], 0])
            visited[i] = True

    while queue:
        trans_word, dist = queue.popleft()

        if trans_word == target:
            answer = dist + 1
            break

        for i in range(word_len):
            if not visited[i] and check_go(trans_word, words[i]):
                queue.append([words[i], dist + 1])
                visited[i] = True

    return answer

begin = "hit"
target = "cog"
words = ["hot", "dot", "dog", "lot", "log", "cog"]
print(solution(begin,target, words))


begin = "hit"
target = "cog"
words = ["hot", "dot", "dog", "lot", "log"]
print(solution(begin,target, words))
```

<br>

#### Mock Exam

```mock exam
def solution(answers):
    answer = []
    give_up = [0,0,0]
    second_rule = [1, 3, 4, 5]
    third_rule = [3, 1, 2, 4, 5]

    for i in range(len(answers)) :
        if answers[i] == (i % 5) + 1 :
            give_up[0] += 1

        if i % 2 == 0:
            if answers[i] == 2:
                give_up[1] += 1
        else:
            if answers[i] == second_rule[(i // 2) % 4]:
                give_up[1] += 1

        if answers[i] == third_rule[(i // 2) % 5]:
            give_up[2] += 1

    for i in range(3) :
        if give_up[i] == max(give_up) :
            answer.append(i + 1)

    return answer


answer = [1,2,3,4,5]
print(solution(answer))

answer = [1,3,2,4,2]
print(solution(answer))
```
<br>

#### Bring Mid Word

```bring mid word
def solution(s):
    answer = ''
    s_len = len(s)
    if s_len % 2 == 0 :
        answer = s[s_len//2 - 1 : s_len//2 + 1]
    else :
        answer = s[s_len//2]
    return answer
```

<br>

#### Make Big Numbers

``` make big numbers
def solution(number, k):
    sub, index, size = 0, 0, len(number)

    while sub != k and size - 1 > index:
        if number[index] != '9' and number[index] < number[index + 1] :
            number = number[:index] + number[index + 1:]
            sub += 1
            index -= 1
            size -= 1
        else :
            index += 1

        if index < 0 :
            index = 0

    return "".join(number[0:size - k + sub])


number = "1924"
k = 3
print(solution(number, k))

number = "1231234"
k = 3
print(solution(number, k))

number = "4177252841"
k = 4
print(solution(number, k))

number = "8888888"
k = 6
print(solution(number, k))
```

<br>

#### Max and Min

``` max and min
def solution(s):
    max_sel, min_sel, arr = -int(1e9), int(1e9), s.split(' ')
    for sel in arr :
        sel_int = int(sel)
        if sel_int > max_sel :
            max_sel = sel_int
        if min_sel > sel_int :
            min_sel = sel_int
    answer = str(min_sel) + " " + str(max_sel)
    return answer


s = "1 2 3 4"
print(solution(s))

s = "-1 -2 -3 -4"
print(solution(s))

s = "-1 -1"
print(solution(s))
```

<br>

#### Priority Queue

``` priority queue
import heapq

def solution(operations):
    answer = []

    for operation in operations:
        oper, num = operation.split(" ")
        if oper == "I":
            heapq.heappush(answer, int(num))
        elif oper == "D" and answer:
            if int(num) > 0:
                answer.pop(len(answer) - 1)
            else:
                heapq.heappop(answer)

    if answer :
        return [max(answer), min(answer)]
    else :
        return [0,0]


operations = ["I 16","D 1"]
print(solution(operations))

operations = ["I 7","I 5","I -5","D -1"]
print(solution(operations))
```

<br>

#### Candidate Key

``` candidate key
from itertools import combinations

def in_check(arr1, arr2) :
    arr1_len = len(arr1)
    result = 0
    for sel in arr1 :
        if sel in arr2 :
            result += 1

    if arr1_len == result :
        return False
    else :
        return True


def solution(relation):
    answer = []
    row, col = len(relation), len(relation[0])
    combination_index, overlap_set = [], []

    for i in range(1, col + 1) :
        combination_index.extend(combinations(range(col), i))

    for combin in combination_index :
        for i in range(row) :
            tup = ()
            for c in combin :
                tup += (relation[i][c],)
            overlap_set.append(tup)

        if len(set(overlap_set)) == row :
            state = True
            for complete in answer :
                state = in_check(complete, combin)
                if not state :
                    break
            if state:
                answer += (combin,)
        overlap_set = []

    return len(answer)


relation = [["100","ryan","music","2"],["200","apeach","math","2"],["300","tube","computer","3"],["400","con","computer","4"],["500","muzi","music","3"],["600","apeach","music","2"]]
print(solution(relation))
```

<br>

#### Best Album

``` best album
def solution(genres, plays):
    answer, genres_len = [], len(genres)
    genres_dict, genres_sum = {}, {}

    for i in range(genres_len):
        if genres[i] in genres_dict.keys() :
            genres_dict[genres[i]].append([plays[i], i])
            genres_sum[genres[i]] += plays[i]
        else :
            genres_dict[genres[i]] = [[plays[i], i]]
            genres_sum[genres[i]] = plays[i]

    for i in sorted(genres_sum, key=lambda x : genres_sum[x], reverse=True) :
        dict_case = sorted(genres_dict[i], key=lambda x : -x[0])
        answer.append(dict_case[0][1])
        if len(dict_case) >= 2:
            answer.append(dict_case[1][1])

    return answer

genres = ["classic", "pop", "classic", "classic", "pop"]
plays = [500, 600, 150, 800, 2500]
print(solution(genres, plays))
```

<br>

#### Travel Route

``` travel route
from collections import defaultdict


def dfs(start, country, visited, answer, tickets_len) :
    answer.append(start)
    if len(answer) == tickets_len :
        return answer

    for i in range(len(country[start])) :
        if visited[start][i] :
            visited[start][i] = False
            if dfs(country[start][i], country, visited, answer, tickets_len) :
                return answer
            visited[start][i] = True

    answer.pop()
    return False


def solution(tickets):
    answer = []
    country, visited, tickets_len = defaultdict(list), defaultdict(list), 0

    for start, destination in tickets:
        country[start].append(destination)
        visited[start].append(True)

    for country_case in country.keys() :
        sort_case = list(sorted(country[country_case]))
        country[country_case] = sort_case
        tickets_len += len(sort_case)

    return dfs("ICN", country, visited, answer, tickets_len + 1)


tickets = [["ICN", "JFK"], ["HND", "IAD"], ["JFK", "HND"]]
print(solution(tickets))

tickets = [["ICN", "SFO"], ["ICN", "ATL"], ["SFO", "ATL"], ["ATL", "ICN"], ["ATL","SFO"]]
print(solution(tickets))

tickets = [["ICN", "AAA"], ["ICN", "BBB"], ["BBB", "ICN"]]
print(solution(tickets))

tickets = [["ICN", "A"], ["ICN", "C"], ["C", "ICN"], ["ICN", "B"], ["B", "ICN"]]
print(solution(tickets))

tickets = [["ICN", "A"], ["A", "B"], ["A", "C"], ["C", "A"], ["B", "D"]]
print(solution(tickets))
```

<br>

#### Fibonachi

```fibonachi
def solution(n):
    fibonachi = [0,1]
    for i in range(2, n + 1) :
         fibonachi.append(fibonachi[i-1] + fibonachi[i-2])
    return fibonachi[n] % 1234567

n = 3
print(solution(n))

n = 5
print(solution(n))
```

<br>

### N's LCM

``` N's LCM
def LCM(sel1, sel2):
    sel1_m = sel1
    sel2_m = sel2
    while sel1_m != sel2_m:
        if sel1_m > sel2_m:
            sel2_m += sel2
        else:
            sel1_m += sel1

    return sel1_m


def solution(arr):
    arr_len = len(arr) - 1

    for i in range(arr_len):
        arr[i + 1] = LCM(arr[i + 1], arr[i])
    return arr[arr_len]


arr = [2,6,8,14]
print(solution(arr))

arr = [1,2,3]   
print(solution(arr))
```

<br>

### Matrix Multiplication

```matrix multiplication
def solution(arr1, arr2):
    answer = []

    for i in range(len(arr1)):
        temporary = []
        for j in range(len(arr2[0])) :
            result = 0
            for k in range(len(arr2)) :
                result += arr1[i][k] * arr2[k][j]
            temporary.append(result)
        answer.append(temporary)
    return answer


arr1 = [[1, 4], [3, 2], [4, 1]]
arr2 = [[3, 3], [3, 3]]
print(solution(arr1, arr2))

arr1 = [[2, 3, 2], [4, 2, 4], [3, 1, 4]]
arr2 = [[5, 4, 3], [2, 4, 1], [3, 1, 1]]
print(solution(arr1, arr2))
```

<br>

#### Cast JadenCase

``` cast JadenCase
def solution(s):
    arr = list(s)
    if 96 < ord(arr[0]) < 123:
        arr[0] = chr(ord(arr[0]) - 32)

    for i in range(1, len(arr)):
        if arr[i - 1] == " " and 96 < ord(arr[i]) < 123:
            arr[i] = chr(ord(arr[i]) - 32)
        elif arr[i - 1] != " " and 64 < ord(arr[i]) < 91 :
            arr[i] = chr(ord(arr[i]) + 32)

    return "".join(arr)


s = "3people UnFolloWeD me"
print(solution(s))

s = "for the last week"
print(solution(s))
```

<br>

#### Land Connect

``` land connect
def find_parent(parent, x):
    if parent[x] != x:
        return find_parent(parent, parent[x])
    return parent[x]


def union_parent(parent, a, b):
    a = find_parent(parent, a)
    b = find_parent(parent, b)

    if a > b:
        parent[a] = b
    else:
        parent[b] = a


def solution(n, costs):
    answer = 0
    parent = [i for i in range(n)]

    costs = sorted(costs, key=lambda x: x[2])

    for case in costs:
        land_1, land_2, cost = case
        if find_parent(parent, land_1) != find_parent(parent, land_2):
            union_parent(parent, land_1, land_2)
            answer += cost

    return answer


n = 4
costs = [[0,1,1],[0,2,2],[1,2,5],[1,3,1],[2,3,8]]
print(solution(n, costs))
```

<br>

#### Longest Palindrome

```longest palindrome
def palindrome_len_standard_even(string, index, string_len):
    if index == 0 :
        return 1

    palindrome, search_len, i = 0, 0, 0
    if string_len // 2 > index:
        search_len = index
    else:
        search_len = string_len - index - 1

    for i in range(1, search_len + 2):
        if index - i < 0 :
            break
        if string[index - i] == string[index + (i - 1)]:
            palindrome += 2
        else:
            break

    return palindrome

def palindrome_len_standard_odd(string, index, string_len):
    palindrome, search_len, i = 1, 0, 0
    if string_len // 2 > index:
        search_len = index
    else:
        search_len = string_len - index - 1

    for i in range(1, search_len + 1):
        if index - i < 0 :
            break
        if string[index - i] == string[index + i]:
            palindrome += 2
        else:
            break

    return palindrome


def solution(s):
    answer = 0
    s_len = len(s)

    for i in range(s_len):
        palindrome_odd = palindrome_len_standard_odd(s, i, s_len)
        palindrome_even = palindrome_len_standard_even(s, i, s_len)

        if palindrome_odd > palindrome_even :
            palindrome = palindrome_odd
        else :
            palindrome = palindrome_even

        if palindrome > answer:
            answer = palindrome
    return answer


s = "abcdcba"
print(solution(s))

s = "abacde"
print(solution(s))

s = "bbb"
print(solution(s))

s = "babbbbbb"
print(solution(s))

s = "bbbaabb"
print(solution(s))
```

<br>

#### Sorting Strings My Own Way

``` sorting strings my own way
def solution(strings, n):
    answer = sorted(strings, key=lambda x : x[n] +x)
    return answer


strings = ["sun", "bed", "car"]
n = 1
print(solution(strings, n))

strings = ["abce", "abcd", "cdx"]
n = 2
print(solution(strings, n))
```

<br>

#### Sorting Strings My Own Way

``` sorting strings my own way
def solution(strings, n):
    answer = sorted(strings, key=lambda x : x[n] +x)
    return answer


strings = ["sun", "bed", "car"]
n = 1
print(solution(strings, n))

strings = ["abce", "abcd", "cdx"]
n = 2
print(solution(strings, n))
```

<br>

#### Jewelry Shopping

``` jewelry shopping
from collections import defaultdict


def solution(gems):
    jewelry_len, gems_len, dic_gems = len(set(gems)), len(gems), defaultdict(int)
    start, end, answer = 0, 0, [0, int(1e9)]

    dic_gems[gems[0]] += 1
    while start < gems_len and end < gems_len :
        if jewelry_len > len(dic_gems.keys()) :
            end += 1
            if end == gems_len :
                break
            dic_gems[gems[end]] += 1
        else :
            if answer[1] - answer[0] > end - start :
                answer = [start, end]
            if dic_gems[gems[start]] == 1 :
                del dic_gems[gems[start]]
            else :
                dic_gems[gems[start]] -= 1
            start += 1

    answer[0] += 1
    answer[1] += 1
    return answer


gems = ["DIA", "RUBY", "RUBY", "DIA", "DIA", "EMERALD", "SAPPHIRE", "DIA"]
print(solution(gems))

gems = ["AA", "AB", "AC", "AA", "AC"]
print(solution(gems))

gems = ["XYZ", "XYZ", "XYZ"]
print(solution(gems))

gems = ["ZZZ", "YYY", "NNNN", "YYY", "BBB"]
print(solution(gems))
```

배운점
 - 검색을 할 때, 딕셔너리를 활용할 수 있는지 생각해보기
 - 배열이 클 때, 완전탐색을 할 수 없다면 while을 활용해 투포인터를 활용할 수 있을지 생각해보기

<br>

#### Shared Taxi Fare

```shared taxi fare
def solution(n, s, a, b, fares):
    min_distance = int(1e9)
    graph = [[int(1e9)] * (n + 1) for _ in range(n + 1)]
    for fare in fares:
        c, d, f = fare
        graph[c][d] = f
        graph[d][c] = f

    for i in range(1, n + 1):
        graph[i][i] = 0

    for k in range(1, n + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                graph[i][j] = min(graph[i][j], graph[i][k] + graph[k][j])

    for i in range(1, n + 1) :
        min_distance = min(graph[s][i] + graph[i][a] + graph[i][b], min_distance)

    return min_distance


n = 6
s = 4
a = 6
b = 2
fares = [[4, 1, 10], [3, 5, 24], [5, 6, 2], [3, 1, 41], [5, 1, 24], [4, 6, 50], [2, 4, 66], [2, 3, 22], [1, 6, 25]]
print(solution(n,s,a,b,fares))

n = 7
s = 3
a = 4
b = 1
fares = [[5, 7, 9], [4, 6, 4], [3, 6, 1], [3, 2, 3], [2, 1, 6]]
print(solution(n,s,a,b,fares))

n = 6
s = 4
a = 5
b = 6
fares = [[2,6,6], [6,3,7], [4,6,7], [6,5,11], [2,5,12], [5,3,20], [2,4,8], [4,3,9]]
print(solution(n,s,a,b,fares))
```

<br>

#### Long Jump

```long jump
def factorial(n) :
    result = 1
    for i in range(1, n + 1) :
        result *= i
    return result


def solution(n):
    answer = 0
    two, one, n_range = 0, 0, n // 2

    for i in range(n_range + 1) :
        two = i
        one = n - (two * 2)

        result = int(factorial(one + two))
        result //= factorial(one)
        result //= factorial(two)
        answer += result

    return answer % 1234567


n = 6
print(solution(n))

n = 5
print(solution(n))

n = 4
print(solution(n))

n = 3
print(solution(n))
```

<br>

#### 2016 Years

``` 2016 years
def solution(a, b):
    day_of_the_week = ["FRI", "SAT", "SUN", "MON", "TUE", "WED", "THU"]
    months = [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days = 0

    for i in range(a) :
        days += months[i]
    days += b
    days = days % 7 - 1

    return day_of_the_week[days % 7]


a = 5
b = 24
print(solution(a, b))
```

<br>

#### Caesar Cipher

``` caesar cipher
def solution(s, n):
    list_s = list(s)

    for i in range(len(list_s)):
        ord_s = ord(list_s[i])
        if (64 < ord_s < 91) or (96 < ord_s < 123):
            trans_s = ord_s + n
        else :
            list_s[i] = ' '
            continue

        if ord_s > 96:
            if trans_s > 122:
                trans_s -= 26
        elif ord_s > 64:
            if trans_s > 90:
                trans_s -= 26

        list_s[i] = chr(trans_s)

    return "".join(list_s)


s = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
n = 25
print(solution(s, n))

s = "abcdefghijklmnopqrstuvwxyz"
n = 25
print(solution(s, n))

s = "a B z"
n = 4
print(solution(s, n))
```

<br>

#### Bad User

``` bad user
import copy


def check_equal_id(id_1, id_2) :
    # id_1은 user_id, id_2는 banned_id
    id_1_len = len(id_1)
    if id_1_len != len(id_2) :
        return False

    for i in range(id_1_len) :
        if id_1[i] != id_2[i] and id_2[i] != '*' :
            return False
    return True


def solution(user_id, banned_id):
    answer, banned_id_len, banned_case, last_arr = 0, len(banned_id), [], []
    candidate_id = [[] for _ in range(banned_id_len)]

    for i in range(banned_id_len) :
        for user in user_id :
            if check_equal_id(user, banned_id[i]) :
                candidate_id[i].append(user)

    for i in candidate_id[0]:
        banned_case.append([i])

    for i in range(1, len(candidate_id)) :
        standard = copy.deepcopy(banned_case)
        for j in standard :
            for k in range(len(candidate_id[i])) :
                copy_j = copy.deepcopy(j)
                if candidate_id[i][k] in copy_j :
                    continue
                else :
                    copy_j.append(candidate_id[i][k])
                banned_case.append(copy_j)
            banned_case.pop(0)

    for i in range(len(banned_case)) :
        arr = set(banned_case[i])
        if arr not in last_arr and len(arr) == banned_id_len :
            last_arr.append(arr)
            answer += 1

    return answer


user_id = ["frodo", "fradi", "crodo", "abc123", "frodoc"]
banned_id = ["fr*d*", "abc1**"]
print(solution(user_id, banned_id))

user_id = ["frodo", "fradi", "crodo", "abc123", "frodoc"]
banned_id = ["*rodo", "*rodo", "******"]
print(solution(user_id, banned_id))

user_id = ["frodo", "fradi", "crodo", "abc123", "frodoc"]
banned_id = ["fr*d*", "*rodo", "******", "******"]
print(solution(user_id, banned_id))

user_id = ["aaaaaa", "bbbbbb", "cccccc", "dddddd", "eeeeee", "ffffff", "gggggg", "hhhhhh"]
banned_id = ["******", "******", "******", "******", "******", "******", "******", "******",]
print(solution(user_id, banned_id))
```

<br>

#### Create Weird Characters

``` create weird characters
def change_alphabat(alphabat) :
    if 64 < ord(alphabat) < 91 :
        return chr(ord(alphabat) + 32)
    elif 96 < ord(alphabat) < 123 :
        return chr(ord(alphabat) - 32)


def solution(s):
    answer = ''
    arr = s.split(" ")
    for sel in range(len(arr)):
        list_sel = list(arr[sel])
        for i in range(len(list_sel)):
            if (i % 2 == 0 and 96 < ord(list_sel[i]) < 123) or (i % 2 == 1 and 64 < ord(list_sel[i]) < 91):
                list_sel[i] = change_alphabat(list_sel[i])
        arr[sel] = "".join(list_sel)

    for i in arr:
        answer += (i + " ")

    return "".join(answer[:-1])


s = "try hello world"
print(solution(s))
```

<br>

#### Skip The Stepping Stone

```skip_the_stepping_stone
def solution(stones, k):
    stones_len = len(stones)
    if sum(stones[0:stones_len // 2]) > sum(stones[stones_len // 2: stones_len]):
        stones.reverse()

    representative_stone = max(stones[0: k])
    min_count = representative_stone

    for i in range(1, stones_len - k + 1):
        if representative_stone == stones[i - 1]:
            if representative_stone > stones[i + k - 1]:
                representative_stone = max(stones[i: i + k])

            elif representative_stone < stones[i + k - 1]:
                representative_stone = stones[i + k - 1]
        else:
            if stones[i + k - 1] > representative_stone:
                representative_stone = stones[i + k - 1]
        if min_count > representative_stone:
            min_count = representative_stone

    return min_count


stones = [2, 4, 5, 3, 2, 1, 4, 2, 5, 1]
k = 3
print(solution(stones, k))

stones = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
k = 3
print(solution(stones, k))

stones = [10,9,8,7,6,5,4,3,2,1]
k = 1
print(solution(stones, k))
```

<br>

#### Hide Phone Number

``` hide phone number
def solution(phone_number):
    answer, phone_number_len = '', len(phone_number)

    for _ in range(phone_number_len - 4):
        answer += "*"

    return answer + phone_number[phone_number_len - 4: phone_number_len + 1]


phone_number = "01033334444"
print(solution(phone_number))

phone_number = "027778888"
print(solution(phone_number))
```

<br>

#### GCD and LCM

``` gcd and lcm
def gcd(n, m) :
    mod = m % n
    if mod != 0 :
        m, n = n, mod
        return gcd(n, m)
    else :
        return n


def solution(n, m) :
    gcd_nm = gcd(n, m)
    return [gcd_nm, int(n*m / gcd_nm)]


n = 5
m = 2
print(solution(n, m))

n = 12
m = 3
print(solution(n, m))
```
