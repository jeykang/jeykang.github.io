
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;

public class Traversal {
    public static <V,E> void DFS( Graph<V,E> g,
                                  Vertex<V> u,
                                  Set<Vertex<V>> known,
                                  Map<Vertex<V>,Edge<E>> forest) {
        //TODO:
        // add u to known
        // for each outgoing edge of u,
        //    v is the opposite of u in the edge
        //    if v is not in known
        //        add e to the forest at v
        //        recursively call DFS with v
        known.add(u);
        for(Edge<E> out: g.outgoingEdges(u)){
			Vertex<V> v = g.opposite(u, out);
			if(!known.contains(v)){
				forest.put(v, out);
				DFS(g, v, known, forest);
			}
		}
    }
    
    public static <V,E> List<Edge<E>> constructPath( Graph<V,E> g,
                                                     Vertex<V> u,
                                                     Vertex<V> v,
                                                     Map<Vertex<V>,Edge<E>> forest ) {
        ArrayList<Edge<E>> path = new ArrayList<>();
        if(forest.get(v) != null) {
            Vertex<V> walk = v;
            while(walk != u) {
                Edge<E> edge = forest.get(walk);
                path.add(0, edge);
                walk = g.opposite(walk, edge);
            }
        }
        return path;
    }
    
    public static <V,E> void BFS( Graph<V,E> g,
                                  Vertex<V> s,
                                  Set<Vertex<V>> known,
                                  Map<Vertex<V>,Edge<E>> forest) {
        Queue<Vertex<V>> queue = new ArrayDeque<>();
        
        //TODO
        // add s to known
        // add s to queue
        // while queue is not empty
        //    remove u from the queue
        //    for each outgoing edge of u
        //        let v be the opposite of u in the edge
        //        if v is not in known
        //           add v to known
        //           add e to forest at v
        //           add v to queue
        known.add(s);
        queue.add(s);
        while(!queue.isEmpty()){
			Vertex<V> u = queue.remove();
			for(Edge<E> out: g.outgoingEdges(u)){
				Vertex<V> v = g.opposite(u, out);
				if(!known.contains(v)){
					known.add(v);
					forest.put(v, out);
					queue.add(v);
				}
			}
		}
    }
    
    protected static void onFalseThrow(boolean b) {
        if(!b)
            throw new RuntimeException("Error: unexpected");
    }         
    public static void main(String[] args) {
        Graph<String,Integer> g = FlightGraph.build();
        Vertex<String> bos = FlightGraph.getAirport(g, "BOS");
        Vertex<String> lax = FlightGraph.getAirport(g, "LAX");

        {
            Set<Vertex<String>> known = new HashSet<Vertex<String>>();
            Map<Vertex<String>, Edge<Integer>> forest = new HashMap<>();
            DFS(g, bos, known, forest); //Depth First Search
            onFalseThrow(known.contains(lax));
    
            for(Edge<Integer> e : constructPath(g, bos, lax, forest))
                System.out.print(e.getElement() + ", ");
            System.out.println();
        }
        
        {
            Set<Vertex<String>> known = new HashSet<Vertex<String>>();
            Map<Vertex<String>, Edge<Integer>> forest = new HashMap<>();
            BFS(g, bos, known, forest); //Breadth First Search
            onFalseThrow(known.contains(lax));
    
            for(Edge<Integer> e : constructPath(g, bos, lax, forest))
                System.out.print(e.getElement() + ", ");
            System.out.println();
        }

        System.out.println("Success!");
        
    }
}
