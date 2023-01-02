import java.util.*;

class Solution {
    public int solution(int[] arrayA, int[] arrayB) {
            int answer = 0;
            int sizeA = arrayA.length;
            int sizeB = arrayB.length;

            int minA = Integer.MAX_VALUE;
            int minB = Integer.MAX_VALUE;

            Set<Integer> set = new HashSet<>();

            for(int i=0;i<sizeA;i++){
                if(minA > arrayA[i]) {
                    minA = arrayA[i];
                }
            }

            for(int i=0;i<sizeB;i++){
                if(minB > arrayB[i]) {
                    minB = arrayB[i];
                }
            }

            for(int i=2;i<=minA;i++){
                if(minA % i == 0){
                    set.add(i);
                }
            }

            for(int i=2;i<=minB;i++){
                if(minB % i == 0){
                    set.add(i);
                }
            }

            int target = 0;
            boolean div = true;
            boolean div_non = true;
            boolean check = true;
            List<Integer> list = new ArrayList<>(set);
            Collections.sort(list, Collections.reverseOrder());

            for(int i=0;i<list.size();i++){
                div = true;
                div_non = true;
                check = true;
                target = list.get(i);
                
                for(int j=0;j<sizeA;j++){
                    if(!div && !div_non){
                        break;
                    }
                    if(arrayA[j] % target == 0){
                        div_non = false;
                    }
                    else{
                        div = false;
                    }
                }
                if(!div && !div_non){
                    continue;
                }
                if(div){
                    for(int k=0;k<sizeB;k++){
                        if(arrayB[k] % target == 0){
                            check = false;
                            break;
                        }
                    }
                    if(check){
                        return target;
                    }
                }
                check = true;
                if(div_non){
                    for(int k=0;k<sizeB;k++){
                        if(arrayB[k] % target != 0){
                            check = false;
                            break;
                        }
                    }
                    if(check){
                        return target;
                    }
                }
            }
            return 0;
        }
}