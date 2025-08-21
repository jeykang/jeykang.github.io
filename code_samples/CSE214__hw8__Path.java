     
    
    public class Path {
        private int[][] dist;
        private World.CellState[][] state;
        private int ny, nx;
        private static int queueType;
        
        public Path(int[][] dist, World.CellState[][] state) {
            this.dist = dist;
            this.state = state;
            ny = state.length;
            nx = state[0].length;
        }
        
        //Prefer dots than empty cells
        protected int cellDist(int x, int y) {
            return state[y][x] == World.CellState.Dot ? 1 : 2;
        }
        
        public void findDistance(Pos from) {
            //For testing, try three different types of queues
            Queue<Pos> queue = null;
            switch(queueType) {
            case 0: queue = new ListQueue<Pos>();   break;
            case 1: queue = new HeapQueue<Pos>();   break;
            case 2: queue = new RBTreeQueue<Pos>(); break;
            }
            queueType = (queueType + 1) % 3;
            queue.enqueue(from);
            //initialize dist to max int
            for(int y = 0; y < ny; y++)
                for(int x = 0; x < nx; x++)
                    dist[y][x] = 998;//Integer.MAX_VALUE;
                    
            dist[from.y][from.x] = 0;
            
            //Update dist array to find the shortest path, where
            //the distance between cells are weighted by cellDist
            //
            //TODO: - Starting from the 'from' position,
            //        update dist[y][x] for all y and x such that
            //        dist[y][x] = dist[p.y][p.x] + d, where
            //        p is the position dequeued from queue and
            //        d is cellDist(x, y) value
            //      - Updating dist[y][x] should continue until
            //        queue  becomes empty
            //
            while(!queue.isEmpty()){
				Pos temp = queue.dequeue();
				for(int i = 0; i < World.DX.length; i++) {
						int x = temp.x + World.DX[i];    //neighbor's x
						int y = temp.y + World.DY[i];    //neighbor's y
						if(0 <= x && x < nx && 0 <= y && y < ny){
							/*if(dist[y][x] > dist[temp.y][temp.x] + 1 && state[y][x] != World.CellState.Wall){
								dist[y][x] = dist[temp.y][temp.x] + 1;
								queue.enqueue(new Pos(x, y));
							}*/
							if(dist[y][x] > dist[temp.y][temp.x] + cellDist(x, y) && state[y][x] != World.CellState.Wall){
								dist[y][x] = dist[temp.y][temp.x] + cellDist(x, y);
								queue.enqueue(new Pos(x, y, ny*y + x));// generate (hopefully) non-overlapping priority for each pos
							}
						}		
                    
				}
			}
    }
    
    //Hopefully, this can help your debugging
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for(int y = 0; y < ny; y++) {
            for(int x = 0; x < nx; x++) {
                if(state[y][x] == World.CellState.Wall)
                    sb.append("###");
                else
                    sb.append(String.format("%3d", dist[y][x]));
            }
            sb.append("\n");
        }
        return sb.toString();
    }
    
    public static void main(String[] args){
        Path path = new Path(new int[60][30], new World.CellState[60][30]);
        for(int i = 0; i < path.state.length - 1; i++){
            for(int j = 0; j < path.state[0].length; j++){
                if((j == i && j % 2 == 0) || (j == i - 4 && j % 3 == 1) || (j == path.state.length - 3 - i && j % 4 == 2) || (i == 13 && j < 10)){
                    path.state[i][j] = World.CellState.Wall;
                }
                else{
                    path.state[i][j] = World.CellState.Dot;
                }
            }
        }
        path.findDistance(new Pos(1, 0, 0));
        System.out.println(path);
    }
}
