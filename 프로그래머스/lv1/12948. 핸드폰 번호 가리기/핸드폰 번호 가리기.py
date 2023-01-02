def solution(phone_number):
    answer, phone_number_len = '', len(phone_number)
    
    for _ in range(phone_number_len - 4) :
        answer += "*"
        
    return answer + phone_number[phone_number_len - 4 : phone_number_len + 1]