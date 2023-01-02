import sys

dic = {"black" : 0, "brown" : 1, "red" : 2, "orange" : 3, "yellow" : 4, "green" : 5, "blue" : 6, "violet" : 7,
       "grey" : 8, "white" : 9}


a = sys.stdin.readline().strip()
b = sys.stdin.readline().strip()
c = sys.stdin.readline().strip()

print(int(str(dic[a]) + str(dic[b])) * (10 ** (dic[c])))