
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

@SuppressWarnings("unchecked")
public class GraphImpl<V,E> implements Graph<V,E>{
    private static class VertexImpl<V,E> implements Vertex<V> {
        private V element;
        private Map<Vertex<V>, Edge<E>> out, in;
        
        public VertexImpl(V elem, boolean isDirected) {
            element = elem;
            out = new HashMap<Vertex<V>, Edge<E>>();
            if(isDirected)
                in = new HashMap<Vertex<V>, Edge<E>>();
            else
                in = out;
        }
        public V getElement()                           { return element; }
        public Map<Vertex<V>, Edge<E>> getOutgoing()    { return out; }
        public Map<Vertex<V>, Edge<E>> getIncoming()    { return in; }
        public static <V,E> VertexImpl<V,E> cast(Vertex<V> v) { return (VertexImpl<V,E>) v; }
    }
    
    private static class EdgeImpl<V,E> implements Edge<E> {
        private E element;
        private Vertex<V>[] endpoints;

        public EdgeImpl(Vertex<V> u, Vertex<V> v, E elem) {
            element = elem;
            endpoints = (Vertex<V>[]) new Vertex[] {u, v};
        }
        public E getElement()               { return element; }
        public Vertex<V>[] getEndpoints()   { return endpoints; }
        public static <V,E> EdgeImpl<V,E> cast(Edge<E> e) { return (EdgeImpl<V,E>) e; }
    }
    
    private boolean isDirected;
    private ArrayList<Vertex<V>> vertices;
    private ArrayList<Edge<E>> edges;

    public GraphImpl() {
        this(false);
    }
    public GraphImpl(boolean isDirected) {
        this.isDirected = isDirected;
        vertices = new ArrayList<Vertex<V>>();
        edges = new ArrayList<Edge<E>>();
    }
    public int numVertices()                            { return vertices.size(); }
    public Iterable<Vertex<V>> vertices()               { return vertices; }
    public int numEdges()                               { return edges.size(); }
    public Iterable<Edge<E>> edges()                    { return edges; }
    public int outDegree(Vertex<V> v)                   { return VertexImpl.cast(v).getOutgoing().size(); }
    public int inDegree(Vertex<V> v)                    { return VertexImpl.cast(v).getIncoming().size(); }
    public Iterable<Edge<E>> outgoingEdges(Vertex<V> v) { return VertexImpl.<V,E>cast(v).getOutgoing().values(); }
    public Iterable<Edge<E>> incomingEdges(Vertex<V> v) { return VertexImpl.<V,E>cast(v).getIncoming().values(); }
    public Edge<E> getEdge(Vertex<V> u, Vertex<V> v)    { return VertexImpl.<V,E>cast(u).getOutgoing().get(v); }
    public Vertex<V>[] endVertices(Edge<E> e)           { return EdgeImpl.<V,E>cast(e).getEndpoints(); }
    public Vertex<V> opposite(Vertex<V> v, Edge<E> e) {
        Vertex<V>[] endp = EdgeImpl.<V,E>cast(e).getEndpoints();
        if(endp[0] == v)        return endp[1];
        else if(endp[1] == v)   return endp[0];
        else throw new IllegalArgumentException("v is not incident to e");
    }
    public Vertex<V> insertVertex(V element) {
        Vertex<V> v = new VertexImpl<V,E>(element, isDirected);
        vertices.add(v);
        return v;
    }
    public Edge<E> insertEdge(Vertex<V> u, Vertex<V> v, E element) {
        if(getEdge(u, v) != null)
            throw new IllegalArgumentException("Edge (u, v) already exists");
        
        EdgeImpl<V,E> e = new EdgeImpl<V,E>(u, v, element);
        edges.add(e);
        VertexImpl.<V,E>cast(u).getOutgoing().put(v, e);
        VertexImpl.<V,E>cast(v).getIncoming().put(u, e);
        return e;
    }
    public void removeVertex(Vertex<V> v) {
        VertexImpl<V,E> vert = VertexImpl.<V,E>cast(v);
        for(Edge<E> e : vert.getOutgoing().values())
            removeEdge(e);
        for(Edge<E> e : vert.getIncoming().values())
            removeEdge(e);
        vertices.remove(v);
    }
    public void removeEdge(Edge<E> e) {
        edges.remove(e);
    }
}
