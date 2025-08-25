/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package recitation4;

/**
 *Assigning Values to Variable
 * The variable binding semantics of for objects and primitives are nearly 
 * identical, but instead of binding a copy of the primitive value, we bind a copy of the object address.
 */
public class Ball {

    public static void main(String args[]) {
        Ball someBall = new Ball();
        System.out.println("Some ball before creating another ball = " + someBall);
        Ball anotherBall = someBall;
        someBall = new Ball();
        System.out.println("Some ball = " + someBall);
        System.out.println("Another ball = " + anotherBall);
    }
}



