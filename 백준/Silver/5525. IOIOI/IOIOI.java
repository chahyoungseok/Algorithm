import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class Main {
    public static List<Integer> kmp(String allString, String pattern){
        int pattern_size = pattern.length();
        int i = 0;
        List<Integer> table = new ArrayList<>();
        for(int k=0;k<pattern_size;k++){
            table.add(0);
        }

        for(int j=1;j<pattern_size;j++){
            while(i>0 && pattern.charAt(i) != pattern.charAt(j)){
                i = table.get(i - 1);
            }
            if(pattern.charAt(i) == pattern.charAt(j)){
                i++;
                table.set(j, i);
            }
        }
        
        i = 0;
        List<Integer> result = new ArrayList<>();
        for(int j=0;j<allString.length();j++){
            while(i>0 && pattern.charAt(i) != allString.charAt(j)){
                i = table.get(i - 1);
            }
            if(pattern.charAt(i) == allString.charAt(j)){
                i++;
                if(i == pattern_size){
                    result.add(j - i + 1);
                    i = table.get(i - 1);
                }
            }
        }
        return result;
    }


    public static void main(String[] args) throws IOException {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        int N = Integer.parseInt(br.readLine());
        String pattern = "I";
        for(int i=0;i<N;i++){
            pattern += "OI";
        }
        int M = Integer.parseInt(br.readLine());

        String S = br.readLine();
        List<Integer> result = kmp(S, pattern);
        System.out.println(result.size());
    }
}