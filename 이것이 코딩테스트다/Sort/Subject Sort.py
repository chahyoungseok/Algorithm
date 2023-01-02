N = int(input())
student_info = []

for _ in range(N) :
    student_info.append(input().split())

student_info.sort(key=lambda x : (-int(x[1]) , int(x[2]), -int(x[3]), x[0]))

for student in student_info :
    print(student[0])