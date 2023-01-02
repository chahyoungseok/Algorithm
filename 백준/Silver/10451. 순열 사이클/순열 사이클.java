import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);
        int T = scan.nextInt();

        for(int t=0;t<T;t++){
            int n = scan.nextInt();
            int edge[] = new int[n + 1];
            boolean visited[] = new boolean[n + 1];
            int count = 0;

            for(int i=1;i<n+1;i++){
                edge[i] = scan.nextInt();
                visited[i] = true;
            }

            for(int i=1;i<n+1;i++){
                if(visited[i]) {
                    dfs(i, edge, visited);
                    count++;
                }
            }
            System.out.println(count);
        }
    }

    public static boolean dfs(int start, int[] edge, boolean[] visited){
        if(!visited[start]){
            return false;
        }
        visited[start] = false;
        if(!dfs(edge[start], edge, visited)){
            return false;
        }
        return true;
    }
}
