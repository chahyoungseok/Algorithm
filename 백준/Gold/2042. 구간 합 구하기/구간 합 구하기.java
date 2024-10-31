import java.io.*;
import java.util.*;

public class Main {

    static long[] elements, tree;

    public static long init(int start, int end, int node) {
        if (start == end) {
            return tree[node] = elements[start];
        }
        int mid = (start + end) / 2;
        return tree[node] = init(start, mid, node * 2) + init(mid + 1, end, node * 2 + 1);
    }

    public static long subSum(int start, int end, int left, int right, int node) {
        if (start > right || end < left) {
            return 0;
        }

        if (left <= start && end <= right) {
            return tree[node];
        }

        int mid = (start + end) / 2;
        return subSum(start, mid, left, right, node * 2) + subSum(mid + 1, end, left, right, node * 2 + 1);
    }

    public static void update(int start, int end, long diff, int index, int node) {
        if (start > index || index > end) {
            return;
        }
        tree[node] += diff;
        if (start != end) {
            int mid = (start + end) / 2;
            update(start, mid, diff, index, node * 2);
            update(mid + 1, end, diff, index, node * 2 + 1);
        }
    }

    public static void main(String[] args) throws IOException {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(System.out));
        StringTokenizer tokenizer = new StringTokenizer(reader.readLine());

        int N = Integer.parseInt(tokenizer.nextToken());
        int M = Integer.parseInt(tokenizer.nextToken());
        int K = Integer.parseInt(tokenizer.nextToken());

        elements = new long[N + 1];
        for (int i=1;i < N + 1;i ++) {
            elements[i] = Long.parseLong(reader.readLine());
        }
        tree = new long[N * 4];
        init(1, N, 1);

        StringBuilder sb = new StringBuilder();
        for (int i=0;i < M+K;i ++) {
            tokenizer = new StringTokenizer(reader.readLine());
            int a = Integer.parseInt(tokenizer.nextToken());
            int b = Integer.parseInt(tokenizer.nextToken());
            long c = Long.parseLong(tokenizer.nextToken());

            if (a == 1) {
                long diff = c - elements[b];
                elements[b] = c;
                update(1, N, diff, b, 1);
            }
            else {
                sb.append(subSum(1, N, b, (int) c, 1) + "\n");
            }
        }

        bw.write(sb.toString());
        bw.flush();
        bw.close();
        reader.close();
    }
}
