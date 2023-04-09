import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

public class Main {
    private static int n, m, a, b;
    private static boolean status = true;
    private static int[] parent;

    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));

        StringTokenizer st = new StringTokenizer(reader.readLine());

        n = Integer.parseInt(st.nextToken());
        m = Integer.parseInt(st.nextToken());

        parent = new int[n];

        for(int i=0;i<n;i++){
            parent[i] = i;
        }

        for(int i=0;i<m;i++){
            st = new StringTokenizer(reader.readLine());

            a = Integer.parseInt(st.nextToken());
            b = Integer.parseInt(st.nextToken());

            if(find_parent(parent, a) != find_parent(parent, b)){
                union_parent(parent, a, b);
            }
            else {
                status = false;
                System.out.println(i + 1);
                break;
            }
        }

        if(status) {
            System.out.println(0);
        }
    }

    public static int find_parent(int[] parent, int x){
        if(parent[x] != x){
            return find_parent(parent, parent[x]);
        }
        return x;
    }

    public static void union_parent(int[] parent, int a, int b){
        a = find_parent(parent, a);
        b = find_parent(parent, b);

        if(a > b) {
            parent[a] = b;
        }
        else {
            parent[b] = a;
        }
    }
}
