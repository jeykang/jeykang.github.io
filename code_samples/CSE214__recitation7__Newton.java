public class Newton {
    public static final double EPS = 1e-10; 
    public interface Func<T, R> {
        public R apply(T a);
    }
    
    public static double newton(Func<Double, Double> f, double x) {
        //TODO: implement the Newton method that finds a root of f
        //      starting from x
        double tempY1 = f.apply(x);
        double tempY2 = f.apply(x + EPS);
        double slope = (tempY2 - tempY1) / EPS;
        double yIcpt = tempY1 - (slope * x); // y intercept
        double xIcpt = -1 * yIcpt / slope; // x intercept
        if(Math.abs(xIcpt - x) < EPS){
            return xIcpt;
        }
        else{
            return newton(f, xIcpt);
        }
    }
    
    public static double sqrt(double x) {
        //TODO: implement sqrt using newton
        return newton(temp -> temp * temp - x, x);
    }
    
    public static void main(String[] args) {
        System.out.println("ans: " + newton(x -> x * x - 2  * x - 5, 10));
        System.out.println("ans: " + sqrt(2));
    }
}