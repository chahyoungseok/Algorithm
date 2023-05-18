import org.w3c.dom.Node;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class Main {

    private static int n, m;
    private static int x, y;
    private static int min_result = (int) 1e9;
    private static int min_host = 0;
    private static int current;
    private static Queue<Node> queue;

    private static ArrayList<ArrayList<Integer>> friends;
    private static boolean[] visited;

    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer st;

        st = new StringTokenizer(reader.readLine());
        friends = new ArrayList<>();

        n = Integer.parseInt(st.nextToken());
        m = Integer.parseInt(st.nextToken());

        visited = new boolean[n + 1];
        for(int i=0;i<n + 1;i++){
            friends.add(new ArrayList<>());
        }

        for(int i=0;i<m;i++){
            st = new StringTokenizer(reader.readLine());

            x = Integer.parseInt(st.nextToken());
            y = Integer.parseInt(st.nextToken());

            friends.get(x).add(y);
            friends.get(y).add(x);
        }

        for(int i=1;i<n+1;i++){
            current = 0;
            queue = new LinkedList<>();
            Arrays.fill(visited, true);

            queue.add(new Node(i, 0));
            visited[i] = false;

            while(!queue.isEmpty()){
                Node next_node = queue.poll();

                for(int j=0;j<friends.get(next_node.node).size();j++){
                    int ca = friends.get(next_node.node).get(j);
                    if(visited[ca]){
                        visited[ca] = false;
                        queue.add(new Node(ca, next_node.dist + 1));
                        current += next_node.dist + 1;
                    }
                }
            }
            if(min_result > current){
                min_result = current;
                min_host = i;
            }
        }

        System.out.println(min_host);
    }

    private static class Node{
        private int node;
        private int dist;

        public Node(int node, int dist) {
            this.node = node;
            this.dist = dist;
        }
    }
}
