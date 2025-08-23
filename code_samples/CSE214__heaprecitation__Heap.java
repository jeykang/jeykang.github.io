 

public class Heap<E extends Comparable<E>> {
    protected E[] arr;
    protected int size;
    
    public Heap() {
        arr = null;
        size = 0;
    }
    public E min() {
        if(size <= 0)
            throw new IndexOutOfBoundsException("Empty heap");
        return arr[0];
    }
    public int size()        { return size; }
    public boolean isEmpty() { return size == 0; }
    
    @SuppressWarnings("unchecked")
    public void add(E e) {
        if(arr == null) {
            arr = (E[]) new Comparable[16];
        }
        //dynamic array
        if(size + 1 == arr.length) {
            E[] tmp = (E[]) new Comparable[arr.length * 2];
            for(int i = 0; i < arr.length; i++)
                tmp[i] = arr[i];
            arr = tmp;
        }
        arr[size] = e;
        size++;
        upheap(size-1);
        
        //TODO: - add e at arr[size], increase size
        //      - call upheap
    }
    public E remove() {
        if(size <= 0)
            throw new IndexOutOfBoundsException("Empty heap");
        E temp = arr[0];
        arr[0] = arr[size-1];
        size--;
        downheap(0);
        
        //TODO: - save arr[0], copy arr[size-1] to arr[0], decrease size
        //      - call downheap
        return temp;
    }
    protected int parent(int i)       { return (i - 1) / 2; }
    protected int left(int i)         { return 2 * i + 1; }
    protected int right(int i)        { return 2 * i + 2; }
    protected void swap(int i, int j) { E tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp; }
    protected void upheap(int i) {
        int p = parent(i);
        if(i == 0 || arr[p].compareTo(arr[i]) <= 0){
            return;
        }
        swap(i, p);
        upheap(p);
        
        //TODO: implement upheap
        // - if i is root or arr[p] <= arr[i], return
        // - swap i and parent
        // - recursively call upheap with parent
    }
    protected void downheap(int i) {
        int l = left(i);
        int r = right(i);
        int c = l;
        if(l >= size){
            return;
        }
        if(r < size && arr[r].compareTo(arr[l]) <= 0) {
            c = r;
        }
        if(arr[c].compareTo(arr[i]) >= 0){
            return;
        }
        swap(i, c);
        downheap(c);
        //TODO: implement downheap
        // - if l >= size, return
        // - if r >= size, c is l
        // - otherwise, c is the smaller of r and l
        // - if arr[c] >= arr[i], return
        // - otherwise, swap i and c and call downheap again
    }
    
    //for the test code
    protected E get(int i) { return arr[i]; }
    
    protected static void onFalseThrow(boolean b) {
        if(!b)
            throw new RuntimeException("Error: unexpected");
    }    
    
    public static void main(String[] args) {
        Heap<Integer> heap = new Heap<Integer>();
        int[] arr = new int[] {3, 5, 2, 4, 1, 8, 7, 6, 0, 9 };
        for(int i = 0; i < 10; i++)
            heap.add(arr[i]);
        
        for(int i = 0; i < 10; i++)
            System.out.print(heap.remove() + ", ");
        System.out.println();
    }
}