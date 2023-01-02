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
