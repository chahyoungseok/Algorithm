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