# Algorithm

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#greedy">Greedy</a></li>
    <li><a href="#day-2">Day 2</a></li>
    <li><a href="#day-3">Day 3</a></li>
    <li><a href="#day-4">Day 4</a></li>
    <li><a href="#day-5">Day 5</a></li>
    <li><a href="#day-6">Day 6</a></li>
    <li><a href="#day-7">Day 7</a></li>
    <li><a href="#day-8">Day 8</a></li>
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
N, M = map(int, input().split())

card = []

for inList in range(0,N) :
    inList = list(map(int,input().split()))
    card.append(min(inList))

print(max(card))
```

<br>

### 1이 될 때까지

``` until it becomes 1

```
