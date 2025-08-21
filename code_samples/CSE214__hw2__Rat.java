 

//Rational number
public class Rat implements Field, Modulo, Ordered {
    private int n, d;
    public Rat(int numerator, int denumerator) {
        Int numGCD = (Int)Euclidean.GCD(new Int(numerator), new Int(denumerator));
        int intGCD = numGCD.getInt();
        n = numerator / intGCD;
        d = denumerator / intGCD;
    }
    public Ring mod(Ring a) {
        Rat r = (Rat)a;
        if(r.n == 0)
            throw new ArithmeticException("Division by zero");
        return new Rat((n*r.d) % (d*r.n), d*r.d);
        }
        //Ordered
    public boolean ge(Ordered a) {
        Rat r = (Rat)a;
        return n*r.d >= d*r.n;
    }
    @Override
    public Ring add(Ring a) {
        // TODO Auto-generated method stub
        Rat ratA = (Rat)a;
        int aNum = ratA.getNum(), aDenum = ratA.getDenum();
        return new Rat((aNum * d + n * aDenum), (d * aDenum));
        
    }
    @Override
    public Ring addIdentity() {
        // TODO Auto-generated method stub
        return new Rat(0, 10);
    }
    @Override
    public Ring addInverse() {
        // TODO Auto-generated method stub
        return new Rat(-n, d);
    }
    @Override
    public Ring mul(Ring a) {
        // TODO Auto-generated method stub
        Rat ratA = (Rat)a;
        int aNum = ratA.getNum(), aDenum = ratA.getDenum();
        return new Rat((n * aNum), (d * aDenum));
    }
    @Override
    public Ring quo(Ring a) {
        // TODO Auto-generated method stub
        Rat ratA = (Rat)a;
        return this.mul(ratA.mulInverse());
    }
    @Override
    public Ring mulIdentity() {
        // TODO Auto-generated method stub
        return new Rat(1, 1);
    }
    @Override
    public Ring mulInverse() throws ArithmeticException {
        // TODO Auto-generated method stub
        return new Rat(d, n);
    }
    public int getNum() {
        return n;
    }
    public int getDenum() {
        return d;
    }
    public String toString(){
        return new String(n+"/"+d);
    }
}
