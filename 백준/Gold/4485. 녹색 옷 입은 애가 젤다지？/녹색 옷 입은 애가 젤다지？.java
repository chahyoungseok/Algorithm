import java.util.Arrays;
import java.util.PriorityQueue;
import java.util.Scanner;
import java.util.Stack;

class Point implements Comparable<Point>{
    private int x;
    private int y;
    private int cost;

    public Point(int x, int y, int cost){
        this.x = x;
        this.y = y;
        this.cost = cost;
    }

    public int getX(){
        return this.x;
    }

    public int getY(){
        return this.y;
    }

    public int getCost(){
        return this.cost;
    }

    @Override
    public int compareTo(Point point) {
        if(this.cost >= point.cost){
            return 1;
        }
        return -1;
    }
}

public class Main {
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);
        int count = 1;
        while(true){
            int N = scan.nextInt();
            if(N == 0){break;}
            int board[][] = new int[N][N];
            int dp[][] = new int[N][N];
            int dx[] = {1, -1, 0, 0};
            int dy[] = {0, 0, 1, -1};

            for(int i=0;i<N;i++){
                for(int j=0;j<N;j++){
                    dp[i][j] = Integer.MAX_VALUE;
                    board[i][j] = scan.nextInt();
                }
            }

            PriorityQueue<Point> priorityQueue = new PriorityQueue<>();
            priorityQueue.add(new Point(0, 0, board[0][0]));
            dp[0][0] = board[0][0];


            while(!priorityQueue.isEmpty()){
                Point point = priorityQueue.poll();
                int x = point.getX();
                int y = point.getY();
                int cost = point.getCost();

                if(x == N - 1 && y == N - 1){
                    continue;
                }

                for(int i=0;i<4;i++){
                    int mx = point.getX() + dx[i];
                    int my = point.getY() + dy[i];
                    if((0<=mx && mx<N) && (0<=my && my<N)){
                        int cost_result = cost + board[mx][my];
                        if(dp[mx][my] > cost_result) {
                            dp[mx][my] = cost_result;
                            priorityQueue.add(new Point(mx, my, cost_result));
                        }
                    }
                }
            }

            System.out.println("Problem " + count + ": " + dp[N - 1][N - 1]);
            count++;
        }
    }
}
