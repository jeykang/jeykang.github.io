/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import java.util.stream.Collectors;

/**
 *
 * @author techj
 */
public class Person {
    public enum Sex{
        MALE, FEMALE
    }
    
    private String name;
    private Integer age;
    private Sex gender;
    private String emailAddress;
    public Person(String name, Integer age, Sex gender, String emailAddress){
        this.name = name;
        this.age = age;
        this.gender = gender;
        this.emailAddress = emailAddress;
    }
    public String getName(){
        return name;
    }
    public Integer getAge(){
        return age;
    }
    public Sex getGender(){
        return gender;
    }
    public String getEmail(){
        return emailAddress;
    }
    public static void lambdaPrintEmailMalesBetweenAge (List<Person> roster, int low, int high) {
        Stream<Person> pStream = roster.stream();
        List<Person> pArray = pStream.filter(p -> p.getAge() < high && p.getAge() > low && p.getGender() == Sex.MALE).collect(Collectors.toList());
        pArray.forEach(p -> System.out.println(p.getEmail()));
    }
    public static void lambdaPrintPersonsOlderThan(List<Person> roster, int age) {
        Stream<Person> pStream = roster.stream();
        List<Person> pArray = pStream.filter(p -> p.getAge() > age).collect(Collectors.toList());
        pArray.forEach(p -> System.out.println(p.getName()));
    }
    public static void lambdaPrintEmailPersonsOlderThan (List<Person> roster, int age) {
        Stream<Person> pStream = roster.stream();
        List<Person> pArray = pStream.filter(p -> p.getAge() > age).collect(Collectors.toList());
        pArray.forEach(p -> System.out.println(p.getEmail()));
    } 
    
    public static void main(String args[]){
        List<Person> roster = new ArrayList<Person>();
        Person mary = new Person("Mary", 20, Sex.FEMALE,
       "mary@stonybrook.edu");
        Person john = new Person("John", 22, Sex.MALE,
       "john@stonybrook.edu");
        Person cogi = new Person("Cogi", 16, Sex.MALE,
       "cogi@stonybrook.edu");
        Person steve = new Person("Steve", 21, Sex.MALE,
       "steve@stonybrook.edu");
        Person bush = new Person("Bush", 19, Sex.MALE,
       "bush@stonybrook.edu");
        Person seoyoung = new Person("Seoyoung", 18, Sex.FEMALE,
       "seoyoung@stonybrook.edu");
        roster.add(mary);
        roster.add(john);
        roster.add(cogi);
        roster.add(steve);
        roster.add(bush);
        roster.add(seoyoung);
        Person.lambdaPrintPersonsOlderThan(roster, 18);
        Person.lambdaPrintEmailPersonsOlderThan(roster, 18);
        Person.lambdaPrintEmailMalesBetweenAge (roster, 16, 20);
     } 
}
