import java.util.*;

class Solution {
    public double[] solution(int k, int[][] ranges) {
        ArrayList<Double> array = new ArrayList<>();
        double[] answer = new double[ranges.length];
        
        double result = 0.0;
        int next_k = 0;
        while(k != 1){
            if(k % 2 == 0){
                next_k = (int) k / 2;
                result = (double)next_k + ((double)(k - next_k) / 2);
            }
            else{
                next_k = k * 3 + 1;
                result = (double)k + ((double)(next_k - k) / 2);
            }
            array.add(result);
            k = next_k;
        }
        int end_point = array.size();
        
        for(int i=0;i<ranges.length;i++){
            int start = ranges[i][0];
            int end = end_point + ranges[i][1];
            double sum = 0.0;
            if (start > end){
                answer[i] = -1.0;
                continue;
            }
            
            for(int j=start;j<end;j++){
                sum += array.get(j);
            }
            answer[i] = sum;
        }
        
        return answer;
    }
}