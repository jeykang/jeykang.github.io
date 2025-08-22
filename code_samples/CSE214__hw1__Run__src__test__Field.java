package src.test;

public interface Field extends Ring {
    public Ring mulIdentity();
    public Ring mulInverse() throws ArithmeticException;
}

