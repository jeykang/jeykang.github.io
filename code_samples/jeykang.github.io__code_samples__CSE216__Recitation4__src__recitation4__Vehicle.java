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
    public class Vehicle {
    private String name;
    public Vehicle(String name) {
        this.name = name;
    }
    public void setName(String name) {
        this.name = name;
    }
    public String getName() {
        return name;
    }
    @Override
    public String toString() {
        return "Vehicle[name = " + name + "]";
    }
}