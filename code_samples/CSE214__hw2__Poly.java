//NOTE: All non-inline comments are failed/changed attempts or debug code.

//import ring.Comp;
//import ring.Field;

@SuppressWarnings("unchecked")
public class Poly<F extends Field> implements Ring, Modulo, Ordered {
    private F[] coef;
    public Poly(F[] coef) {
        int n = coef.length;
        while(n >= 2 && Comp.eq(coef[n-1], coef[0].addIdentity()))
            n--;
        this.coef = (F[])new Field[n];
        for(int i = 0; i < n; i++)
            this.coef[i] = coef[i];
            
    }    
    public F[] getCoef() {
        return coef;
    }
    public String toString() {
        String str = "";
        if(coef.length == 0){
            return new String("empty coef");
        }
        for(int i = coef.length - 1; i > 0; i--)
            str = str + coef[i] + "x^" + i + " + ";
        str = str + coef[0];
        return str;
    }
    public Ring add(Ring a) {
        Poly that = (Poly)a;
        F[] thatCoef = (F[])that.getCoef();
        F[] newCoef, smallerCoef;
        //System.out.println("thiscoef length: "+this.coef.length);
        //System.out.println("thatcoef length: "+thatCoef.length);
        if(this == this.addIdentity()){ //Check to see if one of the vars is addIdentity, and skip the for loop
            return that;
        }
        else if(that == this.addIdentity()){
            return this;
        }
        if(this.coef.length >= thatCoef.length){ //Assign longer and shorter coefs
            newCoef = this.coef.clone();
            smallerCoef = thatCoef.clone();
            /*
            //System.out.println("thiscoef >= thatcoef");
            newCoef = this.coef.clone();
            for(int i = 0; i < thatCoef.length; i++){
                //System.out.println("newCoef before: "+newCoef[i]);
                //System.out.println(this.coef[i]+" added to "+thatCoef[i]+" = "+(F)this.coef[i].add(thatCoef[i]));
                newCoef[i] = (F)this.coef[i].add(thatCoef[i]);
                //System.out.println("newCoef after: "+newCoef[i]);
            }
            */
        }
        else{
            newCoef = thatCoef.clone();
            smallerCoef = this.coef.clone();
            /*
            //System.out.println("thiscoef < thatcoef");
            newCoef = thatCoef.clone();
            for(int i = 0; i < this.coef.length; i++){
                //System.out.println("newCoef before: "+newCoef[i]);
                //System.out.println(this.coef[i]+" added to "+thatCoef[i]+" = "+(F)this.coef[i].add(thatCoef[i]));
                newCoef[i] = (F)this.coef[i].add(thatCoef[i]);
                //System.out.println("newCoef after: "+newCoef[i]);
            }
            */
        }       
        for(int i = 0; i < smallerCoef.length; i++){ //For every element in the shorter coef, add it to the corresponding element in the longer coef
                //System.out.println("newCoef before: "+newCoef[i]);
                //System.out.println(this.coef[i]+" added to "+thatCoef[i]+" = "+(F)this.coef[i].add(thatCoef[i]));                           
                newCoef[i] = (F)newCoef[i].add(smallerCoef[i]);
                //System.out.println("newCoef after: "+newCoef[i]);
            }
        return new Poly(newCoef);
    }
    public Poly del(Poly that) {
        return (Poly)this.add(that.addInverse()); //convenient method to handle adding addInverse for debug
    }
    public Ring mul(Ring a) { //Direct conversion of .mul() from PolyDbl
        Poly that = (Poly)a;
        F[] thatCoef = (F[])(that.getCoef());
        F[] newCoef = (F[])new Field[this.coef.length + thatCoef.length - 1];
        //System.out.println("(mul)this: "+this);
        //System.out.println("(mul)that: "+that);
        //System.out.println("(mul)max index: "+(newCoef.length - 1));
        for(int i = 0; i < this.coef.length; i++){
            for (int j = 0; j < thatCoef.length; j++){
                //System.out.println("(mul)cur i+j: "+(i+j));
                //System.out.println("(mul)thiscoef[i]: "+this.coef[i]);
                //System.out.println("(mul)thatcoef[j]: "+thatCoef[j]);
                //System.out.println("(mul)thiscoef[i] * thatcoef[j]: "+(this.coef[i].mul(thatCoef[j])));
                if(newCoef[i+j] == null){
                    newCoef[i+j] = (F)(this.coef[i].mul(thatCoef[j]));
                }
                else{
                    newCoef[i+j] = (F)newCoef[i+j].add(this.coef[i].mul(thatCoef[j]));            
                }
            }
        }
        return new Poly(newCoef);
    }
    public Poly[] longdiv(Poly that) { //I was originally going to use my own solution for this, but mine used a while loop that was more clunky than the given solution for the recitation
        //return value: longdiv(...)[0]: quotient,
        //              longdiv(...)[1]: remainder
        
        //degree of divisor
        int dd = that.coef.length - 1; 
        
        //quotient
        F[] q = (F[])new Field[this.coef.length - that.coef.length + 1]; 
        
        //remainder
        F[] r = (F[])new Field[this.coef.length];  
        
        //copy coef to r
        for(int i = 0; i < this.coef.length; i++)
        {
            r[i] = this.coef[i];
        }
        
        //the long division algorithm
        for(int qi = q.length-1; qi >= 0; qi--) 
        {
            q[qi] = (F)r[qi + dd].mul(that.coef[dd].mulInverse());
            
            for(int i = 0; i <= dd; i++)
            {
                r[qi+i] = (F)r[qi+i].add((q[qi].mul(that.coef[i])).addInverse());
            }
        }
        
        return new Poly[] {new Poly(q), new Poly(r)};
    }
    public Ring addIdentity() { //Check if there is an element in coef (just for debugging) before returning addIdentity
        if(this.coef.length == 0){
            throw new ArrayIndexOutOfBoundsException();
        }
        
        return new Poly((F[])(new Field[]{(F)this.coef[0].addIdentity()}));
    }
    public Ring addInverse() { //Unlike PolyDbl there is no convenient way to multiply every element by -1
        F[] tempCoef = this.coef.clone();
        for(int i = 0; i < tempCoef.length; i++){
            tempCoef[i] = (F)tempCoef[i].addInverse();
        }
        return new Poly(tempCoef);
    }
    public Ring mod(Ring a) {
        return this.longdiv((Poly)a)[1];
    }//remainder
    public Ring quo(Ring a) {
        return this.longdiv((Poly)a)[0];
    }//quotient
    public boolean ge(Ordered a){   //greater than or equal to
        Poly tempA = (Poly)a;
        F[] aCoef = (F[])(tempA.getCoef());
        if(this.coef.length > aCoef.length){
            ////System.out.println("this length is bigger than that");
            return true;
        }
        else if(this.coef.length == aCoef.length){
            boolean temp = true;
            for(int i = this.coef.length - 1; i >= 0; i--){ //Check if every element is equal
                if(!Comp.eq(this.coef[i], aCoef[i])){
                    temp = false;
                }
            }
            if(temp){ //if every element is same return true
                return true;
            }
            for(int i = this.coef.length - 1; i >= 0; i--){ //Check every coef to see which is bigger
                ////System.out.println("Checking if "+this.coef[i]+" is bigger than "+aCoef[i]);
                if(Comp.lt(this.coef[i], aCoef[i])){
                    ////System.out.println(this+" is smaller than "+tempA);
                    return false;
                }
                else if(Comp.gt(this.coef[i], aCoef[i])){
                    ////System.out.println(this+" is bigger than "+tempA);
                    return true;
                }
            }
        }
        return false;
    }
}