import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class Main {

    private static int n, m, k;
    private static int x1, y1, x2, y2;
    private static int mx, my;
    private static boolean board[][];
    private static int[] dx = {1, -1, 0, 0};
    private static int[] dy = {0, 0, 1, -1};
    private static ArrayList<Integer> results = new ArrayList<>();


    private static Queue<Node> queue;

    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer st;

        st = new StringTokenizer(reader.readLine());

        n = Integer.parseInt(st.nextToken());
        m = Integer.parseInt(st.nextToken());
        k = Integer.parseInt(st.nextToken());

        board = new boolean[m][n];
        for(int i=0;i<m;i++){
            Arrays.fill(board[i], true);
        }

        for(int i=0;i<k;i++){
            st = new StringTokenizer(reader.readLine());

            x1 = Integer.parseInt(st.nextToken());
            y1 = Integer.parseInt(st.nextToken());
            x2 = Integer.parseInt(st.nextToken());
            y2 = Integer.parseInt(st.nextToken());

            for(int j=x1;j<x2;j++){
                for(int o=y1;o<y2;o++){
                    board[j][o] = false;
                }
            }
        }

        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                if(board[j][i]){
                    int size = 1;

                    queue = new LinkedList<>();
                    queue.add(new Node(j, i));
                    board[j][i] = false;

                    while (!queue.isEmpty()){
                        Node next_node = queue.poll();

                        for(int f=0;f<4;f++){
                            mx = next_node.x + dx[f];
                            my = next_node.y + dy[f];
                            if(mx < m && my < n && 0 <= mx && 0 <= my  && !queue.contains(new Node(mx, my)) && board[mx][my]){
                                queue.add(new Node(mx, my));
                                board[mx][my] = false;
                                size += 1;
                            }
                        }
                    }
                    results.add(size);
                }

            }
        }

        Collections.sort(results);

        System.out.println(results.size());
        for(int i=0;i<results.size();i++){
            System.out.print(results.get(i) + " ");
        }
    }

    private static class Node{
        private int x;
        private int y;

        public Node(int x, int y) {
            this.x = x;
            this.y = y;
        }

        @Override
        public boolean equals(Object o){
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            Node node = (Node) o;

            return this.x == node.x && this.y == node.y;
        }
    }
}
