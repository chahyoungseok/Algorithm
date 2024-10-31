import java.io.*;
import java.util.*;

public class Main {

    public static boolean isVPS(String data) {
        Stack<Character> stack = new Stack<>();

        for (char ch : data.toCharArray()) {
            if (ch == '(') {
                stack.push('(');
                continue;
            }

            if (stack.isEmpty()) {
                return false;
            }

            stack.pop();
        }

        return stack.isEmpty();
    }

    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        int T = Integer.parseInt(reader.readLine());

        for (int i=0; i<T; i++) {
            String data = reader.readLine();

            if (isVPS(data)) {
                System.out.println("YES");
            }
            else {
                System.out.println("NO");
            }
        }
    }
}
