/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package recitation6;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Stream;
import javax.swing.JButton;

/**
 *
 * @author SUNY Korea CS
 */
public class Recitation6 {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // Create an ArrayList
        List<Integer> myList = new ArrayList<Integer>();
        myList.add(1);
        myList.add(5);
        myList.add(8);
        // Convert it into a Stream
        Stream<Integer> intStream = myList.stream();

        //Map method
        String[] myArray = new String[]{"bob", "alice", "paul", "ellie"};
        Stream<String> strStream = Arrays.stream(myArray);
        Stream<String> strNewStream = 
             strStream.map(s -> s.toUpperCase());

        //Filter method
        Stream<String> filteredStream = strNewStream
        .filter(s -> s.length() > 4);
        
        //This will result in exception as stream is being reused
        //Use StreamSupplier instead
        //filteredStream.forEach(s -> System.out.println(s));
        
        Supplier<Stream<String>> streamSupplier = () -> Stream.of(myArray);
        streamSupplier.get().forEach(s -> System.out.println(s));
        
        //Filter method
        String[] resultArray = streamSupplier.get()
        .filter(s -> s.length() > 4)
        .toArray(String[]::new);  //notice chaining of Stream operations

        //Lambda expression - iterating over a list and perform some operations
        List<String> pointList = new ArrayList();
        pointList.add("1");
        pointList.add("2");
        pointList.forEach(p ->  {
                            System.out.println(p);
                            //Do more work
                        }
                 );
        
        //Create a new runnable and pass it to a thread
           new Thread(
            () -> {System.out.println("My Runnable");}
            ).start();
        
           //JavaFX GUI event handler
           JButton button =  new JButton("Submit");
            button.addActionListener((e) -> {
                System.out.println("Click event triggered !!");
                e.toString();
            }); 
    }
    
}
