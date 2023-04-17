import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

public class Main {

    private static int N;
    private static int board[][];
    private static boolean visited[];

    private static int minDistance = (int) 1e9;

    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer st;

        N = Integer.parseInt(reader.readLine());
        board = new int[N][N];
        visited = new boolean[N];

        for (int i=0;i<N;i++){
            st = new StringTokenizer(reader.readLine());

            for (int j=0;j<N;j++){
                board[i][j] = Integer.parseInt(st.nextToken());

            }
            visited[i] = true;
        }

        for (int i=0;i<N;i++){
            visited[i] = false;
            dfs(i, i, 0, visited, 0);
            visited[i] = true;
        }

        System.out.println(minDistance);
    }

    public static void dfs(int start, int node, int dist, boolean[] visited, int total){
        if(dist == N - 1 && board[node][start] != 0) {
            minDistance = Math.min(total + board[node][start], minDistance);
            return;
        }

        if(total > minDistance){
            return;
        }

        for (int i=0;i<N;i++){
            if (visited[i] && board[node][i] != 0){
                visited[i] = false;
                dfs(start, i, dist + 1, visited, total + board[node][i]);
                visited[i] = true;
            }
        }
    }
}
