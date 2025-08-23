/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package edu.sunyk.CSE216.interaction;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;
import edu.sunyk.CSE216.preprocessing.preprocessing;
import edu.sunyk.CSE216.statistics.statistics;

/**
 *
 * @author gch04
 */
public class userinterface {
 
    public static void main(String[] args) throws IOException{
        int choice1 = 0;
        int choice2 = 0;
        int choice3 = 0;
        int choice4 = 0;
        int year1 = 0;
        int year2 = 0;
        preprocessing preprocess = new preprocessing("YearlyPopulation.csv");
        statistics statis = new statistics();
        Scanner userInput = new Scanner(System.in);
        String countr = "";
        String init = "start";
        while(true){
            while(init == "start"){
                System.out.println("Welcome! Select one of the following statistical parameters: ");
                System.out.println("1. Mean population");
                System.out.println("2. Median population");
                System.out.println("3. Mode of population");
                System.out.println("4. Standard deviation of population");
                System.out.println("5. Variance of population");
                System.out.println("6. Percentage change in population");
                System.out.println("7. Exit");
                try{
                    System.out.print("User Selects-> ");
                    choice1 = userInput.nextInt();
                    if(choice1 > 0 || choice1 < 7) 
                        init = "country";
                    if(choice1 == 7){
                        System.out.println("GoodBye!");
                        System.exit(0);
                    }
                    else if(choice1 < 1 || choice1 > 7) {
                        System.out.println("Invalid number.");
                        init = "start";
                    }
                }
                catch(Exception e){
                    System.out.println("Invalid input." + e);
                    userInput.next();
                }
            }
            while(init == "country"){
                preprocess.num_country();
                System.out.print("Select country of interest: ");
                try{
                    choice2 = userInput.nextInt();
                    if(choice2 > 259 || choice2 < 1)
                        System.out.println("Invalid value.");
                    else{
                        ArrayList<String> coun = new ArrayList();
                        coun = preprocess.countries();
                        countr = coun.get(choice2 - 1);
                        init = "Year1";
                    }
                }
                catch(Exception e){
                     System.out.println("Invalid input." + e);
                     userInput.next();
                }
            }
            while(init == "Year1"){
                preprocess.ye();
                System.out.print("Select From Year: ");
                try{
                    choice3 = userInput.nextInt();
                    if(choice3 > 58 || choice3 < 1)
                        System.out.println("Invalid input.");
                    else{
                        ArrayList<String> yearList = new ArrayList();
                        yearList = preprocess.year_column();
                        year1 = Integer.parseInt(yearList.get(choice3 - 1));
                        init = "Year2";
                    }
                }
                catch(Exception e){
                    System.out.println("Invalid input.");
                    userInput.next();
                }
            }
            while(init =="Year2"){
                preprocess.ye();
                System.out.print("Select To Year: ");
                try{
                    choice4 = userInput.nextInt();
                    if(choice4 > 58 || choice4 < 1)
                        System.out.println("Invalid input.");
                    else{
                        ArrayList<String> yearList = new ArrayList();
                        yearList = preprocess.year_column();
                        year2 = Integer.parseInt(yearList.get(choice4 - 1));
                        init = "result";
                    }
                }
                catch(Exception e){
                    System.out.println("Invalid input.");
                    userInput.next();
                }
            }
            while(init == "result"){
                double means = statis.meanfromyears(preprocess.dict(), countr, year1, year2);
                double medians = statis.medianfromyears(preprocess.dict(), countr, year1, year2);
                double modes = statis.modefromyears(preprocess.dict(), countr, year1, year2);
                double variances = statis.variancefromyears(preprocess.dict(), countr, year1, year2);
                double percentages = statis.percentchangefromyears(preprocess.dict(), countr, year1, year2);
                String countries = preprocess.country_list(choice2 - 1);
                
                if(choice1 == 1){
                    System.out.println("Output: The mean population of country "+ countries +" from year "+ year1 +" to "+ (year2) +" is "+ means +".");
                }
                else if(choice1 == 2){
                    System.out.println("Output: The median population of country"+ countries +" from year "+ year1 +" to "+ (year2) +" is "+ medians + ".");
                }
                else if(choice1 == 3){
                    System.out.println("Output: The mode population of country "+ countries +" from year "+ year1 +" to "+ (year2) +" is "+ modes + ".");
                }
                else if(choice1 == 4){
                    System.out.println("Output: The Standard deviation of population of country "+ countries +" from year "+ (year2) +" to "+ year2+1 +" is "+ Math.sqrt(variances) + ".");
                }
                else if(choice1 == 5){
                    System.out.println("Output: The Variance of population of country "+ countries +" from year "+ year1 +" to "+ (year2) +" is "+ variances + ".");
                }
                else if(choice1 == 6){
                    System.out.println("Output: Percentage change in population of country "+ countries +" from year "+ year1 +" to "+ (year2) + " is "+ percentages + ".");
                }
                init = "start";
            }
        }
    }
}
