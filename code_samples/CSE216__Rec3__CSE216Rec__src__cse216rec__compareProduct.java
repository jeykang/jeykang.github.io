/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cse216rec;

import java.util.ArrayList;
import java.util.Collections;

/**
 *
 * @author techj
 */
public class compareProduct {
    public static void main(String[] args){
        Product prod1 = new Product("Laptop", 350, "GS65");
        Product prod2 = new Product("Laptop", 350, "Zephyrus S");
        Product prod3 = new Product("Laptop", 90, "FX505");
        Product prod4 = new Product("Monitor", 60, "Test");
        ArrayList<Product> test = new ArrayList();
        test.add(prod3);
        test.add(prod1);
        test.add(prod4);
        test.add(prod2);
        for(Product p: test){
            
            System.out.printf(p+", ");
        }
        System.out.println();
        Collections.sort(test);
        for(Product p: test){
            System.out.printf(p+", ");
        }
    }
}
