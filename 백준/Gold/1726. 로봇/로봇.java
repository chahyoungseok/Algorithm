import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;
import java.util.StringTokenizer;

public class Main {

    private static int M, N;
    private static int s_x, s_y, s_l;
    private static int e_x, e_y, e_l;
    private static Queue<Node> queue;
    private static int[][] board;
    private static boolean[][][] visited;
    private static final int[] dx = {0, 0, 1, 0, -1};
    private static final int[] dy = {0, 1, 0, -1, 0};
    private static final int[] cast_location = {0, 1, 3, 2, 4};


    public static void main(String[] args) throws IOException {
        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));

        StringTokenizer st;

        st = new StringTokenizer(bufferedReader.readLine());

        M = Integer.parseInt(st.nextToken());
        N = Integer.parseInt(st.nextToken());

        board = new int[M][N];
        visited = new boolean[M][N][5];

        for(int i=0;i<M;i++){
            for(int j=0;j<N;j++){
                Arrays.fill(visited[i][j], true);
            }
        }

        for(int i=0;i<M;i++){
            st = new StringTokenizer(bufferedReader.readLine());

            for(int j=0;j<N;j++){
                board[i][j] = Integer.parseInt(st.nextToken());
            }
        }

        st = new StringTokenizer(bufferedReader.readLine());
        s_x = Integer.parseInt(st.nextToken());
        s_y = Integer.parseInt(st.nextToken());
        s_l = Integer.parseInt(st.nextToken());

        st = new StringTokenizer(bufferedReader.readLine());
        e_x = Integer.parseInt(st.nextToken());
        e_y = Integer.parseInt(st.nextToken());
        e_l = Integer.parseInt(st.nextToken());

        queue = new LinkedList<>();
        queue.add(new Node(s_x, s_y, s_l, 0));
        visited[s_x - 1][s_y - 1][s_l] = false;

        int m_x, m_y, m_l;

        while(!queue.isEmpty()){
            Node next_node = queue.poll();

            if(next_node.x == e_x && next_node.y == e_y && next_node.location == e_l){
                System.out.println(next_node.dist);
                break;
            }

            for(int i=1;i<4;i++){
                m_x = next_node.x + (dx[cast_location[next_node.location]] * i);
                m_y = next_node.y + (dy[cast_location[next_node.location]] * i);
                if(i == 1 || i == 2) {
                    if(m_x - 1 >= 0 && m_x - 1 < M && m_y - 1 >= 0 && m_y - 1 < N && board[m_x - 1][m_y - 1] == 0){
                        if(visited[m_x - 1][m_y - 1][next_node.location]){
                            visited[m_x - 1][m_y - 1][next_node.location] = false;
                            queue.add(new Node(m_x, m_y, next_node.location, next_node.dist + 1));
                        }
                    }
                    else {
                        break;
                    }
                }
                else {
                    if(m_x - 1 >= 0 && m_x - 1 < M && m_y - 1 >= 0 && m_y - 1 < N && board[m_x - 1][m_y - 1] == 0
                            && visited[m_x - 1][m_y - 1][next_node.location]){
                        visited[m_x - 1][m_y - 1][next_node.location] = false;
                        queue.add(new Node(m_x, m_y, next_node.location, next_node.dist + 1));
                    }
                }

            }

            m_l = cast_location[next_node.location] + 1;
            if(m_l == 5){
                m_l = 1;
            }
            m_l = cast_location[m_l];
            if(visited[next_node.x - 1][next_node.y - 1][m_l] && board[next_node.x - 1][next_node.y - 1] == 0){
                visited[next_node.x - 1][next_node.y - 1][m_l] = false;
                queue.add(new Node(next_node.x, next_node.y, m_l, next_node.dist + 1));
            }

            m_l = cast_location[next_node.location] - 1;
            if(m_l == 0){
                m_l = 4;
            }
            m_l = cast_location[m_l];
            if(visited[next_node.x - 1][next_node.y - 1][m_l] && board[next_node.x - 1][next_node.y - 1] == 0){
                visited[next_node.x - 1][next_node.y - 1][m_l] = false;
                queue.add(new Node(next_node.x, next_node.y, m_l, next_node.dist + 1));
            }
        }
    }

    private static class Node{
        private int x;
        private int y;
        private int location;
        private int dist;

        public Node(int x, int y, int location, int dist) {
            this.x = x;
            this.y = y;
            this.location = location;
            this.dist = dist;
        }
    }
}
