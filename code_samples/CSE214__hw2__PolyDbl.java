 

public class PolyDbl implements Ring, Modulo, Ordered {
    private double[] coef;
    public PolyDbl(double[] coef) {
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
    public Ring add(Ring a) {
        PolyDbl that = (PolyDbl)a;
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
        return new PolyDbl(newCoef);
    }
    public PolyDbl del(PolyDbl that) { 
        return (PolyDbl)(this.add(that.addInverse()));
    }
    public Ring mul(Ring a) {
        PolyDbl that = (PolyDbl)a;
        double[] thatCoef = that.getCoef();
        double[] newCoef = new double[this.coef.length + thatCoef.length - 1];
        for(int i = 0; i < this.coef.length; i++){
            for (int j = 0; j < thatCoef.length; j++){
                newCoef[i+j] += this.coef[i] * thatCoef[j];
            }
        }
        return new PolyDbl(newCoef);
    }
    public PolyDbl[] longdiv(PolyDbl that) {
        //return value: longdiv(...)[0]: quotient,
        //              longdiv(...)[1]: remainder
        
        //degree of divisor
        int dd = that.coef.length - 1; 
        
        //quotient
        double[] q = new double[this.coef.length - that.coef.length + 1]; 
        
        //remainder
        double[] r = new double[this.coef.length];  
        
        //copy coef to r
        for(int i = 0; i < this.coef.length; i++)
        {
            r[i] = this.coef[i];
        }
        
        //the long division algorithm
        for(int qi = q.length-1; qi >= 0; qi--) 
        {
            q[qi] = r[qi + dd] / that.coef[dd];
            
            for(int i = 0; i <= dd; i++)
            {
                r[qi+i] = r[qi+i] - q[qi] * that.coef[i];
            }
        }
        
        return new PolyDbl[] {new PolyDbl(q), new PolyDbl(r)};
    }
    public Ring addIdentity() {
        return new PolyDbl(new double[]{});
    }
    public Ring addInverse() {
        return this.mul(new PolyDbl(new double[]{-1}));
    }
    public Ring mod(Ring a) {
        return this.longdiv((PolyDbl)a)[1];
    }//remainder
    public Ring quo(Ring a) {
        return this.longdiv((PolyDbl)a)[0];
    }//quotient
    public boolean ge(Ordered a){   //greater than or equal to
        PolyDbl tempA = (PolyDbl)a;
        double[] aCoef = tempA.getCoef();
        PolyDbl tempPoly = this.del(tempA);
        if(tempPoly.getCoef().length == 0){
                return true;
        }
        else if(this.coef.length > aCoef.length){
            return true;
        }
        else if(this.coef.length == aCoef.length){
            if(tempPoly.getCoef()[tempPoly.getCoef().length - 1] > 0){
                return true;
            }
        }
        return false;
    }
}
