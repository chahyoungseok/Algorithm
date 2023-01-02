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