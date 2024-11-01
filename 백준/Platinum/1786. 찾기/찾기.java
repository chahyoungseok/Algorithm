import java.io.*;
import java.util.*;

public class Main {

    public static ArrayList<Integer> kmp(String allString, String pattern) {
        int patternSize = pattern.length();
        int[] table = new int[patternSize];

        int i = 0;
        for (int j = 1; j < patternSize; j++) {
            while (i > 0 && pattern.charAt(i) != pattern.charAt(j)) {
                i = table[i - 1];
            }
            if (pattern.charAt(i) == pattern.charAt(j)) {
                i += 1;
                table[j] = i;
            }
        }

        ArrayList<Integer> result = new ArrayList<>();
        i = 0;
        for (int j = 0; j < allString.length(); j++) {
            while (i > 0 && pattern.charAt(i) != allString.charAt(j)) {
                i = table[i - 1];
            }
            if (pattern.charAt(i) == allString.charAt(j)) {
                i += 1;
                if (i == patternSize) {
                    result.add(j - i + 1);
                    i = table[i - 1];
                }
            }
        }
        return result;
    }

    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));

        String T = reader.readLine();
        String P = reader.readLine();

        ArrayList<Integer> result = kmp(T, P);
        System.out.println(result.size());
        for (Integer number : result) {
            System.out.printf((number + 1) + " ");
        }
    }
}
