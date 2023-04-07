import java.util.*;

public class Main {
    private static int T; //테스트 케이스 수
    private static int N; //컴퓨터 개수
    private static int D; //의존성 개수
    private static int C; //해킹당한 컴퓨터 번호
    private static int[] distances; // 최단거리
    private static List<Edge>[] edgeList;
    private static int cnt = 0; // 총 감염되는 컴퓨터 수
    private static int time = 0; // 마지막 컴퓨터가 감염되기까지의 걸리는 시간

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        T = sc.nextInt();
        for (int t = 0; t < T; t++) {
            N = sc.nextInt();
            D = sc.nextInt();
            C = sc.nextInt();
            distances = new int[N + 1];
            edgeList = new ArrayList[N + 1];
            Arrays.fill(distances, Integer.MAX_VALUE);
            for (int i = 0; i <= N; i++) {
                edgeList[i] = new ArrayList<>();
            }
            for (int i = 0; i < D; i++) {
                int a = sc.nextInt(); //컴퓨터 a
                int b = sc.nextInt(); //컴퓨터 b
                int s = sc.nextInt(); //S초 후 감염
                edgeList[b].add(new Edge(a, s)); // 컴퓨터 b가 감염되면 s초 후 컴퓨터 a도 감염됨을 뜻한다. (a, b 거꾸로 주의)
            }
            distances[C] = 0;
            dijkstra();
            cnt = 0;
            time = 0;
            for (int i = 1; i <= N; i++) {
                if (distances[i] != Integer.MAX_VALUE) {
                    cnt++;
                    time = Math.max(time, distances[i]);
                }
            }
            System.out.println(cnt + " " + time);
        }
    }

    private static void dijkstra() {
        PriorityQueue<Edge> queue = new PriorityQueue<>();
        queue.add(new Edge(C, 0));
        while (!queue.isEmpty()) {
            Edge edge = queue.poll();
            int vertex = edge.vertex;
            int cost = edge.cost;
            if (distances[vertex] < cost) {
                continue;
            }
            for (int i = 0; i < edgeList[vertex].size(); i++) {
                int vertex2 = edgeList[vertex].get(i).vertex;
                int cost2 = edgeList[vertex].get(i).cost + cost;
                if (distances[vertex2] > cost2) {
                    distances[vertex2] = cost2;
                    queue.add(new Edge(vertex2, cost2));
                }
            }
        }
    }

    private static class Edge implements Comparable<Edge> {
        int vertex;
        int cost;

        public Edge(int vertex, int cost) {
            this.vertex = vertex;
            this.cost = cost;
        }

        @Override
        public int compareTo(Edge o) {
            return cost - o.cost;
        }
    }
}