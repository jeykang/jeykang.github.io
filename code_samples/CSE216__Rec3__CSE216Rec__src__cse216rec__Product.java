/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cse216rec;

/**
 *
 * @author techj
 */
public class Product implements Comparable {
    String productCategory;

    int productCost;

    String productName;
    public Product(String cat, int cost, String name){
        productCategory = cat;
        productCost = cost;
        productName = name;
    }

    public int compareTo(Object o) {
        Product p = (Product) o;
        if(this.productCost < p.productCost){
            return -1;
        }
        else if(this.productCost > p.productCost){
            return 1;
        }
        else{
            if(this.productCategory.compareTo(p.productCategory) < 0){
                return -1;
            }
            else if(this.productCategory.compareTo(p.productCategory) > 0){
                return 1;
            }
            else{
                if(this.productName.compareTo(p.productName) < 0){
                    return -1;
                }
                else if(this.productName.compareTo(p.productName) > 0){
                    return 1;
                }
                else{
                    return 0;
                }
            }
        }
        
    }
    public String toString(){
            return productCategory + productCost + productName;
    }
}
