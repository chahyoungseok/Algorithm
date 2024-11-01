import java.io.*;
import java.util.*;

public class Main {

    static int[] parent;

    static class Edge {
        int cost;
        int node;
        int other;
        public Edge(int cost, int node, int other) {
            this.cost = cost;
            this.node = node;
            this.other = other;
        }
    }

    static int findParent(int node) {
        if (node != parent[node]) {
            parent[node] = findParent(parent[node]);
        }
        return parent[node];
    }

    static void unionParent(int node1, int node2) {
        if (node1 == node2) {
            return;
        }

        int parent1 = findParent(node1);
        int parent2 = findParent(node2);
        if (parent1 > parent2) {
            parent[parent1] = parent2;
        }
        else {
            parent[parent2] = parent1;
        }
    }

    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer tokenizer;

        int N = Integer.parseInt(reader.readLine());
        int M = Integer.parseInt(reader.readLine());

        PriorityQueue<Edge> pq = new PriorityQueue<>((edge1, edge2) -> (edge1.cost - edge2.cost));
        for (int i=0; i < M; i++) {
            tokenizer = new StringTokenizer(reader.readLine());
            int a = Integer.parseInt(tokenizer.nextToken());
            int b = Integer.parseInt(tokenizer.nextToken());
            int c = Integer.parseInt(tokenizer.nextToken());

            pq.add(new Edge(c, a, b));
        }

        parent = new int[N + 1];
        for (int i=1; i< N+1; i++) {
            parent[i] = i;
        }

        int result = 0;
        while (!pq.isEmpty()) {
            Edge edge = pq.poll();

            if (findParent(edge.node) == findParent(edge.other)) {
                continue;
            }

            unionParent(edge.node, edge.other);
            result += edge.cost;
        }

        System.out.println(result);
    }
}
