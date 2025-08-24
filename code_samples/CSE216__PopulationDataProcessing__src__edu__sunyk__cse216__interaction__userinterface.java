package edu.sunyk.cse216.interaction;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;
import edu.sunyk.cse216.preprocessing.preprocessing;
import edu.sunyk.cse216.statistics.statistics;

/**
 *
 * @author techj
 */
public class userinterface {

    public static void main(String[] args) throws IOException {
        String state = "Select Param";
        String calc = ""; //what will be calculated
        String country = "";
        Integer fromyear = 0;
        Integer toyear = 0;
        String code = "";
        preprocessing csv = new preprocessing(args[0]);
        statistics calculator = new statistics();
        Scanner input = new Scanner(System.in);
        while (true) {
            while (state.equals("Select Param")) {
                int param = 0;
                System.out.println("Welcome! Select one of the following statistical parameters:\n"
                        + "1. Mean\n"
                        + "2. Median\n"
                        + "3. Mode\n"
                        + "4. Standard Deviation \n"
                        + "5. Variance\n"
                        + "6. Percentage Change\n"
                        + "7. Exit");
                System.out.println("User selects: ");
                try {
                    param = input.nextInt();
                    if (param < 1 || param > 7) {
                        System.out.println("Invalid input!");
                    } else {
                        switch (param) {
                            case 1:
                                calc = "mean in population";
                                state = "Select Country";
                                break;
                            case 2:
                                calc = "median in population";
                                state = "Select Country";
                                break;
                            case 3:
                                calc = "mode in population";
                                state = "Select Country";
                                break;
                            case 4:
                                calc = "standard deviation in population";
                                state = "Select Country";
                                break;
                            case 5:
                                calc = "variance in population";
                                state = "Select Country";
                                break;
                            case 6:
                                calc = "change in population in percent";
                                state = "Select Country";
                                break;
                            case 7:
                                System.exit(0);
                        }
                    }
                } catch (Exception e) {
                    System.out.println("Invalid input!");
                    input.next();
                }

            }
            while (state.equals("Select Country")) {
                int param = 0;
                HashMap<String, String> codedict = csv.codetoname();
                System.out.println("Select country of interest:");
                ArrayList<String> dictkeys = new ArrayList(codedict.keySet());
                Collections.sort(dictkeys);
                for (int i = 0; i < dictkeys.size(); i++) {
                    System.out.println((i + 1) + ". " + dictkeys.get(i));
                }
                System.out.println("User selects: ");
                try {
                    param = input.nextInt();
                    if (param < 1 || param > dictkeys.size()) {
                        System.out.println("Invalid input!");
                    } else {
                        code = dictkeys.get(param - 1);
                        country = codedict.get(code);
                        state = "Select From";
                    }
                } catch (Exception e) {
                    System.out.println("Invalid input!:" + e);
                    input.next();
                }

            }
            while (state.equals("Select From")){
                int param = 0;
                ArrayList<Integer> years = csv.years();
                System.out.println("Select from year:");
                Collections.sort(years);
                for (int i = 0; i < years.size(); i++) {
                    System.out.println((i + 1) + ". " + years.get(i));
                }
                System.out.println("User selects: ");
                try {
                    param = input.nextInt();
                    if (param < 1 || param > years.size()) {
                        System.out.println("Invalid input!");
                    } else {
                        fromyear = years.get(param - 1);
                        state = "Select To";
                    }
                } catch (Exception e) {
                    System.out.println("Invalid input!:" + e);
                    input.next();
                }
            }
            while (state.equals("Select To")){
                int param = 0;
                ArrayList<Integer> years = csv.years();
                System.out.println("Select to year:");
                Collections.sort(years);
                for (int i = 0; i < years.size(); i++) {
                    System.out.println((i + 1) + ". " + years.get(i));
                }
                System.out.println("User selects: ");
                try {
                    param = input.nextInt();
                    if (param < 1 || param > years.size()) {
                        System.out.println("Invalid input!");
                    } else {
                        toyear = years.get(param - 1);
                        state = "Output";
                    }
                } catch (Exception e) {
                    System.out.println("Invalid input!:" + e);
                    input.next();
                }
            }
            if (state.equals("Output")){
                Double calcres = 0.0;
                switch(calc){
                    case "mean in population":
                        //calcres = calculator.mean
                        calcres = calculator.meanfromyears(csv.preprocess(), code, fromyear, toyear);
                        break;
                    case "median in population":
                        //calcres = new Double(calculator.median)
                        calcres = calculator.medianfromyears(csv.preprocess(), code, fromyear, toyear);
                        break;
                    case "mode in population":
                        //calcres = new Double(calculator.mode)
                        calcres = calculator.modefromyears(csv.preprocess(), code, fromyear, toyear);
                        break;
                    case "standard deviation in population":
                        //calcres = calculator.stddev
                        calcres = calculator.stddevfromyears(csv.preprocess(), code, fromyear, toyear);
                        break;
                    case "variance in population":
                        //calcres = calculator.variance
                        calcres = calculator.variancefromyears(csv.preprocess(), code, fromyear, toyear);
                        break;
                    case "change in population in percent":
                        //calcres = calculator.percentchange
                        calcres = calculator.percentchangefromyears(csv.preprocess(), code, fromyear, toyear);
                        break;        
                }
                System.out.println("The "+calc+" of country "+country+" from year "+fromyear.toString()+" to "+toyear.toString()+" is "+calcres.toString());
                state = "Select Param";
            }
        }
    }
}
