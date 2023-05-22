import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class Main {
    private static int N, M;
    private static int size = 0, max_size = 0, info = 0, count = 0;
    private static int mx, my;
    private static int[][] board;
    private static boolean[][] visited;
    private static int[][] family;
    private static ArrayList<Integer> family_size;
    private static int[] dx = {1, 0, -1, 0};
    private static int[] dy = {0, 1, 0, -1};
    private static int[] switching = {8, 4, 2, 1};
    private static Queue<Node> queue;

    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer st;

        st = new StringTokenizer(reader.readLine());
        N = Integer.parseInt(st.nextToken());
        M = Integer.parseInt(st.nextToken());

        board = new int[M][N];
        visited = new boolean[M][N];
        queue = new LinkedList<>();

        family = new int[M][N];
        family_size = new ArrayList<>();
        family_size.add(0);


        for(int i=0;i<M;i++){
            st = new StringTokenizer(reader.readLine());

            for(int j=0;j<N;j++){
                board[i][j] = Integer.parseInt(st.nextToken());
                visited[i][j] = true;
                family[i][j] = -1;
            }
        }

        int family_spec = 0;

        for(int i=0;i<M;i++){
            for(int j=0;j<N;j++){
                if(visited[i][j]){
                    size = 0;

                    count++;
                    family_spec++;
                    visited[i][j] = false;
                    queue.add(new Node(i, j));

                    while(!queue.isEmpty()){
                        Node next_node = queue.poll();

                        info = board[next_node.x][next_node.y];
                        family[next_node.x][next_node.y] = family_spec;
                        size++;

                        for(int k=0;k<4;k++){
                            if (info >= switching[k]){
                                info -= switching[k];
                            }
                            else{
                                mx = next_node.x + dx[k];
                                my = next_node.y + dy[k];

                                if(mx >= 0 && mx < M && my >= 0 && my < N && visited[mx][my] && !queue.contains(new Node(mx, my))){
                                    queue.add(new Node(mx, my));
                                    visited[mx][my] = false;
                                }
                            }
                        }
                    }
                    family_size.add(size);
                    max_size = Math.max(max_size, size);
                }
            }
        }

        int two_room_max = 0;
        int spec = -1;

        for(int i=0;i<M;i++) {
            for (int j=0;j<N;j++) {
                spec = family[i][j];

                for(int k=0;k<4;k++){
                    mx = i + dx[k];
                    my = j + dy[k];

                    if(mx >= 0 && mx < M && my >= 0 && my < N && spec != family[mx][my]){
                        two_room_max = Math.max(two_room_max, family_size.get(spec) + family_size.get(family[mx][my]));
                    }
                }
            }
        }
        
        System.out.println(count);
        System.out.println(max_size);
        System.out.println(two_room_max);
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
