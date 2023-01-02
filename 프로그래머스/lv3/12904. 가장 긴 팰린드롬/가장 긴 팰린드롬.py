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