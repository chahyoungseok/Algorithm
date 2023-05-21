import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class Main {

    private static int N, M;
    private static int A, B, C;
    private static int result;
    private static int last_cost;
    private static int[] parents;
    private static PriorityQueue<Info> priorityQueue;

    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer st;

        st = new StringTokenizer(reader.readLine());
        priorityQueue = new PriorityQueue<>();
        result = 0;

        N = Integer.parseInt(st.nextToken());
        M = Integer.parseInt(st.nextToken());

        parents = new int[N + 1];
        for(int i=0;i<N + 1;i++){
            parents[i] = i;
        }

        for(int i=0;i<M;i++){
            st = new StringTokenizer(reader.readLine());

            A = Integer.parseInt(st.nextToken());
            B = Integer.parseInt(st.nextToken());
            C = Integer.parseInt(st.nextToken());

            priorityQueue.add(new Info(A, B, C));
        }

        while(!priorityQueue.isEmpty()){
            Info info = priorityQueue.poll();
            if(find_parent(parents, info.a) != find_parent(parents, info.b)){
                union_parent(parents, info.a, info.b);
                result += info.cost;
                last_cost = info.cost;
            }
        }

        System.out.println(result - last_cost);
    }

    public static int find_parent(int[] parents, int x){
        if(parents[x] != x){
            return find_parent(parents, parents[x]);
        }
        return x;
    }

    public static void union_parent(int[] parents, int a, int b){
        a = find_parent(parents, a);
        b = find_parent(parents, b);

        if(a > b){
            parents[a] = b;
        }
        else {
            parents[b] = a;
        }
    }

    private static class Info implements Comparable<Info>{
        private int a;
        private int b;
        private int cost;

        public Info(int a, int b, int cost) {
            this.a = a;
            this.b = b;
            this.cost = cost;
        }

        @Override
        public int compareTo(Info info) {
            if(this.cost > info.cost){
                return 1;
            }
            else {
                return -1;
            }
        }
    }
}
