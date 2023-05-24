import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

public class Main {

    private static final int maxValue = (int) 1e9;
    private static int minValue = (int) 1e9;
    private static boolean isCircle = false;
    private static int V, E;
    private static int a, b, c;
    private static int[][] edges;

    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer st;

        st = new StringTokenizer(reader.readLine());

        V = Integer.parseInt(st.nextToken());
        E = Integer.parseInt(st.nextToken());

        edges = new int[V+1][V+1];
        for(int i=0;i<V+1;i++){
            Arrays.fill(edges[i], maxValue);
        }

        for(int i=0;i<E;i++){
            st = new StringTokenizer(reader.readLine());

            a = Integer.parseInt(st.nextToken());
            b = Integer.parseInt(st.nextToken());
            c = Integer.parseInt(st.nextToken());

            edges[a][b] = c;
        }

        for(int k=1;k<V+1;k++){
            for(int i=1;i<V+1;i++){
                for(int j=1;j<V+1;j++){
                    edges[i][j] = Math.min(edges[i][j], edges[i][k] + edges[k][j]);
                }
            }
        }

        for(int i=1;i<V+1;i++){
            for(int j=1;j<V+1;j++){
                if(edges[i][j] + edges[j][i] > maxValue){
                    continue;
                }
                minValue = Math.min(minValue, edges[i][j] + edges[j][i]);
                isCircle = true;
            }
        }

        if(isCircle){
            System.out.println(minValue);
        }
        else{
            System.out.println(-1);
        }
    }
}
