package src.test;

public class Int implements Ring, Modulo, Ordered {
	private int n;
	private static Int addId;
	static {
		addId = new Int(0);
	}
	public Int(int n) { this.n = n; }
	public int getInt() { return n; }
	//Ring
	public Ring add(Ring a) { return new Int(n + ((Int)a).n); }
	public Ring addIdentity() { return addId; }
	public Ring addInverse() { return new Int(-n); }
	public Ring mul(Ring a) { return new Int(n * ((Int)a).n); }
	//Modulo
	public Ring mod(Ring a) { return new Int(n % ((Int)a).n); }
	public Ring quo(Ring a) { return new Int(n / ((Int)a).n); }
	//Ordered
	public boolean ge(Ordered a) { return n >= ((Int)a).n; }
}
