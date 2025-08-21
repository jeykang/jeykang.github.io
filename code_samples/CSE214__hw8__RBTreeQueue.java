 

public class RBTreeQueue<E extends Comparable<E>> implements Queue<E> {
    private RBTree<E> rbTree;
    
    public RBTreeQueue() {
        rbTree = new RBTree<E>();
    }
    public int size() {
        //TODO: implement this method
        return rbTree.size();
    }
    public boolean isEmpty() {
        //TODO: implement this method
        return rbTree.size() == 0;
    }
    public void enqueue(E e) {
        //TODO: implement this method
        rbTree.insert(e);
    }
    public E dequeue() {
        //TODO: implement this method
        return rbTree.delete(rbTree.min());
    }
    public E first() {
        //TODO: implement this method
        return rbTree.min();
    }
    public static void main(String[] args){
		RBTreeQueue test = new RBTreeQueue();
		test.enqueue("a");
		test.enqueue("b");
		test.enqueue("c");
		test.enqueue("d");
		test.enqueue("e");
		while(!test.isEmpty()){
			System.out.println(test.first());
			System.out.println(test.dequeue());
		}
	}
}
