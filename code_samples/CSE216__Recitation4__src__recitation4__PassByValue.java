/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package recitation4;

/**
 *
 * @author SUNY Korea CS
 */
public class PassByValue {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        PassByValue byValue = new PassByValue();
        int someValue = 7;
        System.out.println("Before calling method (value = " + someValue + ")");
        byValue.process(someValue);
        System.out.println("After calling method (value = " + someValue + ")");
    }
    
    public void process(int value) {
    System.out.println("Entered method (value = " + value + ")");
    value = 50;
    System.out.println("Changed value within method (value = " + value + ")");
    System.out.println("Leaving method (value = " + value + ")");
}

    
}
