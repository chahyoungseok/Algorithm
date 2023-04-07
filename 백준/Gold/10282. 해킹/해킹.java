import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.PriorityQueue;
import java.util.StringTokenizer;

public class Main {

    private static int n, d, c;
    private static ArrayList<ArrayList<Node>> graph;
    private final static int INF = (int) 1e9;
    private static int[] distances = new int[10001];


    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));

        int t = Integer.parseInt(reader.readLine());

        StringTokenizer st;

        StringBuilder sb = new StringBuilder();

        while (t-- > 0) {
            st = new StringTokenizer(reader.readLine());

            n = Integer.parseInt(st.nextToken()); // 노드의 개수
            d = Integer.parseInt(st.nextToken()); // 간선의 개수
            c = Integer.parseInt(st.nextToken()); // 시작점

            graph = new ArrayList<>();

            for (int i=0;i<n+1;i++){
                graph.add(new ArrayList<>());
            }

            for (int i=0;i<d;i++){
                st = new StringTokenizer(reader.readLine());

                int a = Integer.parseInt(st.nextToken()); // 의존자
                int b = Integer.parseInt(st.nextToken()); // 의존주
                int s = Integer.parseInt(st.nextToken()); // 시간

                graph.get(b).add(new Node(a, s));
            }

            Arrays.fill(distances, INF);

            dijkstra(c);

            int count = 0;
            int result = 0;

            for(int i=1;i<n+1;i++){
                if(distances[i] != INF){
                    count++;
                    result = Math.max(result, distances[i]);
                }
            }
            sb.append(count + " " + result + "\n");
        }

        System.out.println(sb);
    }

    private static void dijkstra(int start){
        PriorityQueue<Node> q = new PriorityQueue<>();
        q.add(new Node(start, 0));
        distances[start] = 0;

        while (!q.isEmpty()) {
            Node next_node = q.poll();
            int dist = next_node.getDistance();
            int now = next_node.getNode();

            if (dist > distances[now]){
                continue;
            }

            for (Node next : graph.get(now)){
                int cost = next.getDistance() + dist;

                if (distances[next.getNode()] > cost){
                    distances[next.getNode()] = cost;
                    q.add(new Node(next.getNode(), cost));
                }
            }
        }
    }

    public static class Node implements Comparable<Node>{
        private int node;
        private int distance;

        public Node(int node, int distance) {
            this.node = node;
            this.distance = distance;
        }

        public int getNode() {
            return this.node;
        }

        public int getDistance() {
            return this.distance;
        }

        @Override
        public int compareTo(Node other) {
            return this.distance - other.distance;
        }
    }
}
