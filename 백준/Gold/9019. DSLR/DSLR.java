import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class Main {

    private static int t, a, b;
    private static Queue<Node> queue;
    private static boolean visited[];

    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer st;

        t = Integer.parseInt(reader.readLine());
        for(int i=0;i<t;i++){
            st = new StringTokenizer(reader.readLine());
            queue = new LinkedList<>();

            visited = new boolean[10000];
            for(int j=0;j<10000;j++){
                visited[j] = true;
            }

            a = Integer.parseInt(st.nextToken());
            b = Integer.parseInt(st.nextToken());

            queue.add(new Node(a, ""));
            visited[a] = false;

            while (!queue.isEmpty()) {
                Node next_node = queue.poll();

                int D_result = D(next_node.result);
                int S_result = S(next_node.result);
                int L_result = L(next_node.result);
                int R_result = R(next_node.result);

                if(D_result == b) {
                    System.out.println(next_node.input + "D");
                    break;
                }
                if(S_result == b) {
                    System.out.println(next_node.input + "S");
                    break;
                }
                if(L_result == b) {
                    System.out.println(next_node.input + "L");
                    break;
                }
                if(R_result == b) {
                    System.out.println(next_node.input + "R");
                    break;
                }

                if(visited[D_result]){
                    visited[D_result] = false;
                    queue.add(new Node(D_result, next_node.input + "D"));
                }

                if(visited[S_result]){
                    visited[S_result] = false;
                    queue.add(new Node(S_result, next_node.input + "S"));
                }

                if(visited[L_result]){
                    visited[L_result] = false;
                    queue.add(new Node(L_result, next_node.input + "L"));
                }

                if(visited[R_result]){
                    visited[R_result] = false;
                    queue.add(new Node(R_result, next_node.input + "R"));
                }
            }
        }
    }

    private static int D(int input){
        if(input * 2 >= 10000) {
            return (input * 2) % 10000;
        }
        return input * 2;
    }

    private static int S(int input){
        if(input == 0) {
            return 9999;
        }
        return input - 1;
    }

    private static int L(int input){
        String temp = String.valueOf(input);
        int temp_length = temp.length();

        StringBuilder sb = new StringBuilder();

        for(int i=0;i<4-temp_length;i++){
            sb.append("0");
        }
        sb.append(temp);

        return Integer.parseInt(sb.substring(1) + sb.charAt(0));
    }

    private static int R(int input){
        String temp = String.valueOf(input);
        int temp_length = temp.length();

        StringBuilder sb = new StringBuilder();

        for(int i=0;i<4-temp_length;i++){
            sb.append("0");
        }
        sb.append(temp);

        return Integer.parseInt(sb.charAt(3) + sb.substring(0, 3));
    }

    private static class Node {
        private int result;
        private String input;

        public Node(int result, String input) {
            this.result = result;
            this.input = input;
        }
    }
}
