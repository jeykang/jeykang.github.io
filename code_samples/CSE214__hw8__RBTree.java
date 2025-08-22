 

import java.util.ArrayList;

public class RBTree<E extends Comparable<E>> {
    public static class Node<E extends Comparable<E>> {
        public E e;
        public Node<E> parent, left, right;
        public boolean isRed;
        public Node() {
            this(null, false, null, null, null);
        }
        public Node(E e, boolean isRed, Node<E> p, Node<E> l, Node<E> r) {
            this.e = e;
            this.isRed = isRed; parent = p; left = l; right = r;
        }
    }
    
    protected Node<E> root;
    protected int size;
    
    public RBTree() {
        root = null;
        size = 0;
    }
    public int size()        { return size; }
    public boolean isEmpty() { return size == 0; }
    public Node<E> root()    { return root; }
    public E min()           { return minNode(root).e; }
    public E max()           { return maxNode(root).e; }
    public E insert(E e) {
        if(isEmpty()) {
            root = new Node<E>(e, false/*isRed*/, null, null, null);
            size++;
            return e;
        }
        
        Node<E> node = findNodeOrFutureParent(root, e);
        int cmp = e.compareTo(node.e);
        if(cmp == 0) {      //key exists; replace
            E ret = node.e;
            node.e = e;
            return ret;
        }
        else {              //not found; add e
            Node<E> fresh = new Node<E>(e, true/*isRed*/, null, null, null);
            setParent(fresh, node, cmp < 0/*asleft*/);
            size++;
            rebalanceInsert(fresh);
            return e;
        }
    }
    public E delete(E e) {
        Node<E> node = findNodeOrFutureParent(root, e);
        if(e.compareTo(node.e) != 0)    //not found
            return null;

        //return value in node.e
        E ret = node.e;
        
        //if node is not external, switch with its predecessor/successor 
        if(node.left != null) {
            Node<E> predecessor = maxNode(node.left);
            node.e = predecessor.e;
            node = predecessor;
        }
        else if(node.right != null) {
            Node<E> successor = minNode(node.right);
            node.e = successor.e;
            node = successor;
        }
        
        //now, node has at most 1 child
        if(node.left != null || node.right != null) {
            //promoted child
            Node<E> promote = node.left != null ? node.left: node.right;
            deleteNode(node);
            if(!node.isRed)
                rebalanceDelete(promote);
        }
        else {
            if(!node.isRed)
                rebalanceDelete(node);
            deleteNode(node);
        }

        return ret;
    }
    protected boolean isInternal(Node<E> node) { return node.left != null || node.right != null; }
    protected boolean isRed(Node<E> node)      { return node != null && node.isRed; }
    protected boolean isBlack(Node<E> node)    { return !isRed(node); }
    protected Node<E> minNode(Node<E> root)    { return root.left  == null ? root: minNode(root.left); }
    protected Node<E> maxNode(Node<E> root)    { return root.right == null ? root: maxNode(root.right); }
    protected Node<E> sibling(Node<E> node) {
        return node.parent == null ? null
           :   node.parent.left == node ? node.parent.right
           :   node.parent.left
           ;
    }
    protected Node<E> findNodeOrFutureParent(Node<E> root, E e) {
        int cmp = e.compareTo(root.e);
        if(cmp == 0) 
            return root;
        else if(cmp < 0 && root.left != null)
            return findNodeOrFutureParent(root.left, e);
        else if(cmp < 0 && root.left == null)
            return root;
        else if(cmp > 0 && root.right != null)
            return findNodeOrFutureParent(root.right, e);
        else //if(cmp > 0 && root.right == null)
            return root;
    }
    
    protected static final boolean AS_LEFT_CHILD = true;
    protected static final boolean AS_RIGHT_CHILD = false;
    protected void setParent(Node<E> node, Node<E> parent, boolean asleft) {
        if(node != null)
            node.parent = parent;
        if(parent != null) {
            if(asleft)  parent.left  = node;
            else             parent.right = node;
        }
    }
    
    protected void rotateLeft(Node<E> x){
		Node<E> node = x.parent;
        Node<E> y = node.right;
        node.right = y.left;
        if(y.left != null){
            y.left.parent = node;
        }
        if(node.parent == null)
            this.root = y;
        else if(node == node.parent.left){
            node.parent.left = y;
        }
        else{
            node.parent.right = y;
        }
        y.left = node;
        y.parent = node.parent;
        node.parent = y;
    }
    
    protected void rotateRight(Node<E> x){
		Node<E> node = x.parent;
        Node<E> y = node.left;
        node.left = y.right;
        if(y.right != null){
            y.right.parent = node;
        }
        if(node.parent == null)
            this.root = y;
        else if(node == node.parent.left){
            node.parent.left = y;
        }
        else{
            node.parent.right = y;
        }
        y.right = node;
        y.parent = node.parent;
        node.parent = y;
    }
    
    protected void rotate(Node<E> node) {
        Node<E> x = node;
        Node<E> y = x.parent;   //assumed to be exist
        Node<E> z = y.parent;   //may be null
        
        //if(z == null){
        if(y.right == x){
            rotateLeft(x);
        }
        else{
            rotateRight(x);
        }
        //}
        /*else{
            if(z.right == y && y.right == x){
                rotateLeft(y);
            }
            else if(z.left == y && y.left == x){
                rotateRight(y);
            }
            else if(z.left == y && y.right == x){
                rotateRight(x);
                rotateLeft(x);
            }
            else{
                rotateLeft(x);
                rotateRight(x);
            }
        }*/
        //TODO: implement this method
    }
    
    protected Node<E> restructure(Node<E> node) {
        Node<E> x = node;
        Node<E> y = x.parent;
        Node<E> z = y.parent;
        
        //TODO: implement this method
        if((z.right == y && y.right == x) || (z.left == y && y.left == x)){
            rotate(y);
            return y;
        }
        else{
            rotate(x);
            rotate(x);
            return x;
        }
    }
    
    protected void fixDoubleRed(Node<E> node) {
        if(!(node.isRed && node.parent.isRed))
            return;
        Node<E> parent = node.parent;
        Node<E> uncle = sibling(parent);
        
        //TODO: implement this method
        //      case 1: malformed 4 node
        //      case 2: overflow
        if(uncle == null || !uncle.isRed){
            Node base = restructure(node);
            base.isRed = false;
            base.left.isRed = true;
			base.right.isRed = true;
        }
        else if(uncle != null){
            parent.isRed = false;
            uncle.isRed = false;
            if(parent.parent != this.root){
                parent.parent.isRed = true;
                fixDoubleRed(parent.parent);
            }
        }
        
    }

    protected void rebalanceInsert(Node<E> node) {
        if(node != root) {
            node.isRed = true;
            fixDoubleRed(node);
        }
    }
    
    protected void fixDoubleBlack(Node<E> node) {
        Node<E> z = node.parent;
        Node<E> y = sibling(node);
        boolean zWasRed = isRed(z);
        //TODO: implement this method
        //case 1: transfer
        if(y == null) return;
        Node<E> x;
        if(isRed(y.left)){
			x = y.left;
		}
		else if(isRed(y.right)){
			x = y.right;
		}
		else{
			x = new Node();
		}
        if(isBlack(y) && isRed(x)){
            Node base = restructure(x);
            base.isRed = zWasRed;
            base.left.isRed = false;
            base.right.isRed = false;
        }   
        //case 2: fusion
        else if(isBlack(y) && isBlack(y.left) && isBlack(y.right)){
            y.isRed = true;
            z.isRed = false;
            if(!zWasRed){
                fixDoubleBlack(z);
            }
        }
        //case 3: re-orientation
        else if(isRed(y)){
            rotate(y);
            y.isRed = false;
            z.isRed = true;
            fixDoubleBlack(node);
        }
    }
    
    // node is a promoted child of a deleted node or
    // a to be deleted node if it is external
    protected void rebalanceDelete(Node<E> node) {
        if(node.isRed)              //deleted node was black 
            node.isRed = false;     //regain the black depth
        else if(node != root)
            fixDoubleBlack(node);   //regain the black depth
    }
    
    protected E deleteNode(Node<E> node) {
        if(node.left != null && node.right != null)
            throw new IllegalArgumentException("has 2 children");
        
        Node<E> child = node.left != null ? node.left : node.right;
        if(child != null)
            child.parent = node.parent;
        if(node == root)
            root = child;
        else 
            setParent(child, node.parent, node == node.parent.left/*asleft*/);
        size--;
        return node.e;
    }
    
    protected void preorderRecur(Node<E> node, ArrayList<Node<E>> snapshot) {
        snapshot.add(node);
        if(node.left != null)
            preorderRecur(node.left, snapshot);
        if(node.right!= null)
            preorderRecur(node.right, snapshot);
    }
    
    public Iterable<Node<E>> preorder() {
        ArrayList<Node<E>> snapshot = new ArrayList<>();
        preorderRecur(root, snapshot);
        return snapshot;
    }
    
    protected static void onFalseThrow(boolean b) {
        if(!b)
            throw new RuntimeException("Error: unexpected");
    }    
    public static void main(String[] args) {
        RBTree<Integer> rbtree = new RBTree<>();
        
        //increasing order
        for(int i = 0; i < 10; i++){
            rbtree.insert(i);
		}
        /*test*/ {
            int[] num = new int[] { 3, 1, 0, 2, 5, 4, 7, 6, 8, 9 };
            boolean[] red = new boolean[10];
            red[6] = red[9] = true;
            int k = 0;
            for(Node<Integer> n : rbtree.preorder()) {
                System.out.println("Insert test "+n.e);
                onFalseThrow(n.e == num[k]);
                onFalseThrow(n.isRed == red[k++]);
            }
        }
        
        for(int i = 0; i < 5; i++)
            rbtree.delete(i);
        /*test*/ {
            int[] num = new int[] { 7, 5, 6, 8, 9 };
            boolean[] red = new boolean[10];
            red[2] = red[4] = true;
            int k = 0;
            for(Node<Integer> n : rbtree.preorder()) {
				System.out.println("Delete test "+n.e);
				System.out.println("Target "+num[k]);
				System.out.println("color is red? "+n.isRed);
                onFalseThrow(n.e == num[k]);
                onFalseThrow(n.isRed == red[k++]);
            }
        }
        for(int i = 5; i < 10; i++)
            rbtree.delete(i);
        System.out.println("Success: increasing order");
        
        
        //decreasing order
        for(int i = 9; i >= 0; i--)
            rbtree.insert(i);
        /*test*/ {
            int[] num = new int[] { 6, 4, 2, 1, 0, 3, 5, 8, 7, 9 };
            boolean[] red = new boolean[10];
            red[2] = red[4] = true;
            int k = 0;
            for(Node<Integer> n : rbtree.preorder()) {
                onFalseThrow(n.e == num[k]);
                onFalseThrow(n.isRed == red[k++]);
            }
        }
        
        for(int i = 9; i >= 5; i--)
            rbtree.delete(i);
        /*test*/ {
            int[] num = new int[] {2, 1, 0, 4, 3 };
            boolean[] red = new boolean[10];
            red[2] = red[4] = true;
            int k = 0;
            for(Node<Integer> n : rbtree.preorder()) {
                onFalseThrow(n.e == num[k]);
                onFalseThrow(n.isRed == red[k++]);
            }
        }
        for(int i = 4; i >= 0; i--)
            rbtree.delete(i);
        System.out.println("Success: decreasing order");

        
        //random order
        int[] arr = new int[] {3, 5, 2, 4, 1, 8, 7, 6, 0, 9 };
        for(int i = 0; i < 10; i++)
            rbtree.insert(arr[i]);
        /*test*/ {
            int[] num = new int[] { 5, 3, 1, 0, 2, 4, 7, 6, 8, 9 };
            boolean[] red = new boolean[10];
            red[1] = red[3] = red[4] = red[6] = red[9] = true;
            int k = 0;
            for(Node<Integer> n : rbtree.preorder()) {
                onFalseThrow(n.e == num[k]);
                onFalseThrow(n.isRed == red[k++]);
            }
        }
        
        arr = new int[] {1, 4, 2, 3, 9, 6, 7, 5, 0, 8 };
        for(int i = 0; i < 5; i++)
            rbtree.delete(arr[i]);
        /*test*/ {
            int[] num = new int[] { 5, 0, 7, 6, 8 };
            boolean[] red = new boolean[10];
            red[2] = true;
            int k = 0;
            for(Node<Integer> n : rbtree.preorder()) {
                onFalseThrow(n.e == num[k]);
                onFalseThrow(n.isRed == red[k++]);
            }
        }
        for(int i = 5; i < 10; i++)
            rbtree.delete(arr[i]);
        System.out.println("Success: random order");
    }
}
  
