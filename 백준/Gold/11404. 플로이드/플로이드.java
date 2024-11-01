import java.io.*;
import java.util.*;

public class Main {

    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer tokenizer;

        int n = Integer.parseInt(reader.readLine());
        int m = Integer.parseInt(reader.readLine());
        int inf = Integer.MAX_VALUE;

        int[][] board = new int[n+1][n+1];
        for (int i=0; i < n+1; i++) {
            Arrays.fill(board[i], inf);
            board[i][i] = 0;
        }

        for (int i=0; i < m; i++) {
            tokenizer = new StringTokenizer(reader.readLine());
            int a = Integer.parseInt(tokenizer.nextToken());
            int b = Integer.parseInt(tokenizer.nextToken());
            int c = Integer.parseInt(tokenizer.nextToken());

            board[a][b] = Math.min(board[a][b], c);

        }

        for (int k=1; k < n+1; k++) {
            for (int i=1; i < n+1; i++) {
                for (int j=1; j < n+1; j++) {
                    if (board[i][k] == inf || board[k][j] == inf) {
                        continue;
                    }
                    board[i][j] = Math.min(board[i][j], board[i][k] + board[k][j]);
                }
            }
        }

        for (int i=1; i < n+1; i++) {
            for (int j=1; j < n+1; j++) {
                if (board[i][j] == inf) {
                    System.out.printf("0 ");
                }
                else {
                    System.out.printf(board[i][j] + " ");
                }
            }
            System.out.println();
        }
    }
}
