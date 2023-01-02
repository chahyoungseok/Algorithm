import java.util.HashMap;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);
        HashMap<String, Integer> hashMap = new HashMap<>();
        String N = scan.nextLine();
        int sum = 0;
        for(int i=0;i<10;i++){
            hashMap.put(String.valueOf(i), 0);
        }

        for(int i=0;i<N.length();i++){
            String target = String.valueOf(N.charAt(i));
            sum += Integer.parseInt(target);
            hashMap.put(String.valueOf(target), hashMap.get(String.valueOf(target)) + 1);
        }

        String result = "";
        if(sum % 3 == 0 && hashMap.get("0") != 0){
            for(int i=9;i>-1;i--){
                for(int j=0;j<hashMap.get(String.valueOf(i));j++) {
                    result += String.valueOf(i);
                }
            }
            System.out.println(result);
        }
        else {
            System.out.println(-1);
        }
    }
}