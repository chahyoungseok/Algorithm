import java.io.*;
import java.util.*;

public class Main {

    static int N;
    static int M;
    static int inf = Integer.MAX_VALUE;
    static long[] distances;
    static ArrayList<Node> edges;

    static class Node {
        int nowNode;
        int nextNode;
        int cost;

        public Node(int nowNode, int nextNode, int cost) {
            this.nowNode = nowNode;
            this.nextNode = nextNode;
            this.cost = cost;
        }
    }

    public static boolean bellmanFord(int start) {
        distances[start] = 0;

        for (int i = 1; i < N+1; i++) {
            for (int j = 0; j < M; j++) {
                Node node = edges.get(j);

                if (distances[node.nowNode] != inf && distances[node.nextNode] > distances[node.nowNode] + node.cost) {
                    distances[node.nextNode] = distances[node.nowNode] + node.cost;
                    if (i == N) {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer tokenizer = new StringTokenizer(reader.readLine());

        N = Integer.parseInt(tokenizer.nextToken());
        M = Integer.parseInt(tokenizer.nextToken());

        edges = new ArrayList<>();
        for (int i = 0; i < M; i++) {
            tokenizer = new StringTokenizer(reader.readLine());
            int A = Integer.parseInt(tokenizer.nextToken());
            int B = Integer.parseInt(tokenizer.nextToken());
            int C = Integer.parseInt(tokenizer.nextToken());

            edges.add(new Node(A, B, C));
        }

        distances = new long[N + 1];
        Arrays.fill(distances, inf);
        if (bellmanFord(1)) {
            System.out.println("-1");
        } else {
            for (int i = 2; i < N + 1; i++) {
                if (distances[i] == inf) {
                    System.out.println("-1");
                } else {
                    System.out.println(distances[i]);
                }
            }
        }
        reader.close();
    }
}
