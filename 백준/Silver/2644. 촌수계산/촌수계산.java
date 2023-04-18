import org.w3c.dom.Node;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class Main {

    private static int n, a, b, m, child, parent, result;
    private static List<List<Integer>> graph;
    private static Queue<Node> queue;
    private static boolean[] visited;

    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer st;
        queue = new LinkedList<>();
        result = -1;

        n = Integer.parseInt(reader.readLine());
        graph = new ArrayList<>();
        visited = new boolean[n+1];

        for(int i=0;i<n+1;i++){
            graph.add(new ArrayList<>());
            visited[i] = true;
        }

        st = new StringTokenizer(reader.readLine());
        a = Integer.parseInt(st.nextToken());
        b = Integer.parseInt(st.nextToken());

        m = Integer.parseInt(reader.readLine());
        for(int i=0;i<m;i++){
            st = new StringTokenizer(reader.readLine());
            parent = Integer.parseInt(st.nextToken());
            child = Integer.parseInt(st.nextToken());

            graph.get(parent).add(child);
            graph.get(child).add(parent);
        }

        queue.add(new Node(a, 1));
        visited[a] = false;

        while(!queue.isEmpty() && result == -1) {
            Node next_node = queue.poll();
            int node = next_node.node;
            int dist = next_node.dist;

            for(int nd : graph.get(node)){
                if(visited[nd]){
                    visited[nd] = false;

                    if(nd == b){
                        result = dist;
                        break;
                    }
                    queue.add(new Node(nd, dist + 1));
                }
            }


        }

        System.out.println(result);
    }

    private static class Node {
        private int node;
        private int dist;

        public Node(int node, int dist) {
            this.node = node;
            this.dist = dist;
        }

        public int getNode() {
            return node;
        }

        public int getDist() {
            return dist;
        }
    }
}
