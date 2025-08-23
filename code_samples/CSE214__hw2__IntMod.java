 

//Integer modulo m (m is a prime number)
public class IntMod extends UnitTest implements Field, Ordered {
    private int n, m;
    private Object mulInv;	
	public IntMod(int n, int m) {
		if(m <= 0) {
			throw new IllegalArgumentException("Not a positive divisor");
		}
		n = n % m;
		n = n < 0 ? n + m : n;
		this.n = n;
		this.m = m;
		this.mulInv = null;
	}
	public String toString(){
	    return new String(n+"%"+m);
	}

	public Ring mulInverse() throws ArithmeticException {
		//TODO: find and return x such that ( x * n ) % m = 1
		for (int x = 0; x <= m; x++){ 
				if ((x * n) % m == 1) {
					return new IntMod(x, m);
				}
		}
		return new IntMod(1, 2);
	}
	public int getInt() {
		return n;
	}
	public int getM() {
	    return m;
	   }
    public Ring mulIdentity() {
    	return new IntMod(1, m);
    }
	public Ring add(Ring a) {
		// TODO Auto-generated method stub
		IntMod newA = (IntMod)a;
		int newN = n + newA.getInt(), newM = newA.getM();
		if (newM != m) {
		    throw new IllegalArgumentException("The modulos are not equal");
		}
		return new IntMod(newN, m);
	}
	public Ring addIdentity() {
		// TODO Auto-generated method stub
		return new IntMod(0, m);
	}
	public Ring addInverse() {
		// TODO Auto-generated method stub
		return new IntMod(m - n, m);
	}
	public Ring mul(Ring a) {
		// TODO Auto-generated method stub
		IntMod newA = (IntMod)a;
		int multiple = this.n * newA.getInt(), newM = newA.getM();
		return new IntMod(multiple, newM);
	}
	public boolean ge(Ordered a) {
		// TODO Auto-generated method stub
		IntMod newA = (IntMod)a;
		if (this.n >= newA.getInt()) {
			return true;
		}
		return false;
	}
}

