/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package edu.sunyk.cse216.preprocessing;

import java.util.*;
import com.opencsv.CSVReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/**
 *
 * @author techj
 */
public class preprocessing {

    HashMap<String, List> csvData = new HashMap();
    CSVReader popData;
    List<String[]> popAll;

    public preprocessing(String popPath) throws FileNotFoundException, IOException {
        popData = new CSVReader(new FileReader(popPath));
        popAll = popData.readAll();

    }

    public HashMap<String, ArrayList<Double[]>> preprocess() throws IOException {

        String[] topline = popAll.get(0);
        HashMap<String, ArrayList<Double[]>> popdict = new HashMap<>();
        for (int i = 0; i < popAll.size(); i++) {
            if (i > 0) {
                ArrayList<String> popline = new ArrayList(Arrays.asList(popAll.get(i)));
                popdict.put(popline.get(1), new ArrayList<Double[]>());
                for (int j = 0; j < popline.subList(2, popline.size()).size(); j++) {
                    //System.out.println("Entered loop");
                    if (topline[j + 2].compareTo("") != 0 && popline.get(j + 2).compareTo("") != 0) {
                        popdict.get(popline.get(1)).add(new Double[]{Double.parseDouble(topline[j + 2]), Double.parseDouble(popline.get(j + 2))});
                    } else if (popline.get(j + 2).compareTo("") == 0) {
                        popdict.get(popline.get(1)).add(new Double[]{Double.parseDouble(topline[j + 2]), new Double(0)});
                    }
                }
            }
        }
        return popdict;
    }

    public HashMap<String, String> codetoname() {
        HashMap<String, String> namedict = new HashMap<>();
        for (int i = 0; i < popAll.size(); i++) {
            if (!popAll.get(i)[1].equals("Country Code")) {
                namedict.put(popAll.get(i)[1], popAll.get(i)[0]);
            }
        }
        return namedict;
    }

    public ArrayList<Integer> years() {
        ArrayList<String> temp = new ArrayList(Arrays.asList(popAll.get(0)).subList(2, popAll.get(0).length));
        ArrayList<Integer> result = new ArrayList();
        for (int i = 0; i < temp.size(); i++) {
            result.add(Integer.parseInt(temp.get(i)));
        }
        return result;
    }

    public static void main(String[] args) throws IOException {
        preprocessing test = new preprocessing("YearlyPopulation.csv");
        HashMap testing = test.preprocess();
        Iterator iterator = testing.keySet().iterator();

        while (iterator.hasNext()) {
            String key = iterator.next().toString();
            String value = testing.get(key).toString();

            System.out.println(key + " " + value);
        }
    }
}
