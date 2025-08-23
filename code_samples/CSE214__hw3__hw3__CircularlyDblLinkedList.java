 

import java.util.Iterator;

public class CircularlyDblLinkedList<E> implements List<E>, Iterable<E> {
    protected static class Node<E> {
        public E e;
        public Node<E> prev, next;
        public Node() { 
            this.e = null; this.prev = this; this.next = this;
        }
        public Node(E e, Node<E> prev, Node<E> next) {
            this.e = e; this.prev = prev; this.next = next;
        }
    }
    public static class NodeIterator<E> implements Iterator<E> {
        private Node<E> head, curr;
        public NodeIterator(Node<E> head, Node<E> curr) {
            this.head = head; this.curr = curr;
        }
        //TODO: implement Iterator<E>
        public boolean hasNext(){
            if(curr.next == head){
                return false;
            }
            return true;
        }
        
        public E next(){
            if(hasNext()){
                curr = curr.next;
                return (E)curr.e;
            }
            else{
                throw new ArrayIndexOutOfBoundsException("There are no elements left in this list");
            }
        }
    }
    
    protected Node<E> head;
    protected int size;

    //constructor
    public CircularlyDblLinkedList() {
        head = new Node<E>(); 
        size = 0;
    }
    
    //toString for debugging
    public String toString(){
        String temp = "[";
        for(int i = 0; i < size; i++){
            temp += get(i) + ", ";
        }
        temp += "]";
        return temp;
    }
    
    //TODO: implement interface List
    public E get(int i) {
        return findNode(i).e;
    }
    
    public int size(){
        return size;
    }
    
    public boolean isEmpty(){
        if(head.next == head){
            return true;
        }
        return false;
    }
    
    public E set(int i, E e){
        Node temp = findNode(i);
        Node newNode = new Node(e, temp.prev, temp.next);
        temp.prev.next = newNode;
        temp.next.prev = newNode;
        return (E)temp.e;
    }
    
    public void add(int i, E e){
        Node temp = head.next;
        for(int j = 0; j < i; j++){
            temp = temp.next;
        }
        Node newNode = new Node(e, temp.prev, temp);
        temp.prev.next = newNode;
        temp.prev = newNode;
        size++;
        endPointer();
    }
    
    public E remove(int i){
        Node temp = findNode(i);
        temp.prev.next = temp.next;
        temp.next.prev = temp.prev;
        size--;
        endPointer();
        return (E)temp.e;
    }
    //TODO: implement interface Iterable
    public Iterator<E> iterator(){
        return new NodeIterator(head, head);
    }

    //helper methods
    protected Node<E> findNode(int i) {
        if(i < 0 || i >= size)
            throw new IndexOutOfBoundsException("invalid index: " + i + " is not in [ 0, " + size + ")");
        Node temp = head.next;
        for(int j = 0; j < i; j++){
            temp = temp.next;
        }
        return temp;
    }
    /*
    public boolean isIn(E e) {
        Node temp = head;
        for(int i = 0; i < size; i++){
            temp = temp.next;
            if(e.compareTo(temp.e) == 0){
                return true;
            }
        }
        return false;
    }
    */
   
    public void endPointer(){
        if(size > 0){
            Node tempNode = findNode(size - 1);
            tempNode.next = head;
            head.prev = tempNode;
        }
    }
}
