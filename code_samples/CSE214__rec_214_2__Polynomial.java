public class Polynomial {
    private double[] coef;
    
    public Polynomial(double[] coef) {
        //trim the leading zeros
        int n = coef.length;
        while(n-1 >= 0 && coef[n-1] == 0)
            n--;
        
        this.coef = new double[n];
        for(int i = 0; i < n; i++)
            this.coef[i] = coef[i];
    }
    public double[] getCoef() {
        return coef;
    }
    public String toString() {
        String str = "";
        for(int i = coef.length - 1; i > 0; i--)
            str = str + coef[i] + "x^" + i + " + ";
        str = str + coef[0];
        return str;
    }
    public Polynomial add(Polynomial that) {
        double[] thatCoef = that.getCoef();
        double[] newCoef, smallerCoef;
        if(this.coef.length >= thatCoef.length){
            newCoef = this.coef.clone();
            smallerCoef = thatCoef.clone();
        }
        else{
            newCoef = thatCoef.clone();
            smallerCoef = this.coef.clone();
        }
        for(int i = 0; i < smallerCoef.length; i++){
            newCoef[i] += smallerCoef[i];
        }
        return new Polynomial(newCoef);
    }
    public void del(Polynomial that) { 
        double[] thatCoef = that.getCoef();
        for(int i = 0; i < thatCoef.length; i++){
            thatCoef[i] *= -1;
        }
        this.add(new Polynomial(thatCoef));
    }
    public Polynomial mul(Polynomial that) {
        double[] thatCoef = that.getCoef();
        double[] newCoef = new double[this.coef.length + thatCoef.length - 1];
        for(int i = 0; i < this.coef.length; i++){
            for (int j = 0; j < thatCoef.length; j++){
                newCoef[i+j] += this.coef[i] * thatCoef[j];
            }
        }
        return new Polynomial(newCoef);
    }
    public Polynomial[] longdiv(Polynomial that) {
        
        Polynomial[] result = new Polynomial[2];//[0]=quotient, [1]=remainder
        result[0] = new Polynomial(new double[]{0});
        result[1] = new Polynomial(this.getCoef());
        double[] divisorCoef = that.getCoef(),remainderCoef = result[1].getCoef();
        while(result[1].getCoef() != new double[]{0} && result[1].getCoef().length >= that.getCoef().length){
            double[] tempCoef = new double[result[1].getCoef().length-that.getCoef().length];
            tempCoef[tempCoef.length-1] = remainderCoef[remainderCoef.length-1] / divisorCoef[divisorCoef.length-1];
            Polynomial tempPoly = new Polynomial(tempCoef);
            result[0].add(tempPoly);
            result[1].del(tempPoly.mul(that));
            System.out.println("delete: "+tempPoly.mul(that));
            System.out.println("remainder: "+result[1]);
        }
        
        return result;
    }
    public static void main(String[] args) {
        Polynomial a = new Polynomial(new double[] {-1, 1});
        System.out.println("a: " + a);
        
        Polynomial b = new Polynomial(new double[] { 1, 1});
        System.out.println("b: " + b);
        
        Polynomial test = new Polynomial(a.getCoef());
        test.del(b);
        System.out.println("test: "+test);
        
        Polynomial c = a.add(b);
        System.out.println("c = (a + b): " + c);
        
        Polynomial d = a.mul(b);
        System.out.println("d = (a * b): " + d);
        
        Polynomial e = d.add(c);
        System.out.println("e = (d + c): " + e);
        
        Polynomial[] f = e.longdiv(a);
        System.out.println("e / a: " + f[0]);
        System.out.println("e % a: " + f[1]);
    }
}