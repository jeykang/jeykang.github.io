/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package edu.sunyk.cse216.statistics;

import java.util.*;
import java.util.stream.*;

/**
 *
 * @author techj
 */
public class statistics {
    public Double mean(ArrayList<Double> lst){
        Stream<Double> temp = lst.stream();
        return temp.reduce((x, y) -> x + y).orElse(-1.0) / new Double(lst.size());
    }
    public Double median(ArrayList<Double> lst){
        Collections.sort(lst);
        Stream<Double> temp = lst.stream();
        return temp.filter(x -> x == lst.get(lst.size()/2 - 1) || x == lst.get(lst.size()/2)).reduce((x, y) -> x+y).orElse(0.0) / 2;
    }
    public Double mode(ArrayList<Double> lst){
        Stream<Double> temp = lst.stream();
        final ArrayList<Double> sortedCount = new ArrayList();
        lst.forEach(x -> sortedCount.add(x));
        ArrayList<Double> newCount = new ArrayList(sortedCount.stream().sorted().collect(Collectors.toList()));
        return temp.filter(x -> Collections.frequency(lst, x) == sortedCount.get(sortedCount.size()-1)).collect(Collectors.toList()).get(0);
    }
    public Double variance(ArrayList<Double> lst){
        Stream<Double> temp = lst.stream();
        ArrayList<Double> tempsum = new ArrayList();
        temp.forEach(x -> tempsum.add(Math.pow((x - mean(lst)), 2)));
        Stream<Double> sumstream = tempsum.stream();
        return sumstream.reduce((x, y) -> x + y).orElse(-1.0) / (lst.size()-1);
    }
    //percentchange
    public Double percentchange(ArrayList<Double> lst){
        return lst.stream().reduce((x, y) -> (y - x)/x).orElse(-1.0) * 100;
    }
    //meanfromyears
    public Double meanfromyears(HashMap<String, ArrayList<Double[]>> popdict, String code, Integer start, Integer end){
        ArrayList<Double[]> temp = new ArrayList(popdict.get(code).stream().filter(x -> x[0] < end + 1 && x[0] >= start).collect(Collectors.toList()));
        ArrayList<Double> temp2 = new ArrayList(temp.stream().flatMap(x -> Arrays.stream(new Double[]{x[1]})).collect(Collectors.toList()));
        return mean(temp2);
    }
    //medianfromyears
    public Double medianfromyears(HashMap<String, ArrayList<Double[]>> popdict, String code, Integer start, Integer end){
        ArrayList<Double[]> temp = new ArrayList(popdict.get(code).stream().filter(x -> x[0] < end + 1 && x[0] >= start).collect(Collectors.toList()));
        ArrayList<Double> temp2 = new ArrayList(temp.stream().flatMap(x -> Arrays.stream(new Double[]{x[1]})).collect(Collectors.toList()));
        return median(temp2);
    }
    //modefromyears
    public Double modefromyears(HashMap<String, ArrayList<Double[]>> popdict, String code, Integer start, Integer end){
        ArrayList<Double[]> temp = new ArrayList(popdict.get(code).stream().filter(x -> x[0] < end + 1 && x[0] >= start).collect(Collectors.toList()));
        ArrayList<Double> temp2 = new ArrayList(temp.stream().flatMap(x -> Arrays.stream(new Double[]{x[1]})).collect(Collectors.toList()));
        return mode(temp2);
    }
    //variancefromyears
    public Double variancefromyears(HashMap<String, ArrayList<Double[]>> popdict, String code, Integer start, Integer end){
        ArrayList<Double[]> temp = new ArrayList(popdict.get(code).stream().filter(x -> x[0] < end + 1 && x[0] >= start).collect(Collectors.toList()));
        ArrayList<Double> temp2 = new ArrayList(temp.stream().flatMap(x -> Arrays.stream(new Double[]{x[1]})).collect(Collectors.toList()));
        return variance(temp2);
    }
    //stddevfromyears
    public Double stddevfromyears(HashMap<String, ArrayList<Double[]>> popdict, String code, Integer start, Integer end){
        ArrayList<Double[]> temp = new ArrayList(popdict.get(code).stream().filter(x -> x[0] < end + 1 && x[0] >= start).collect(Collectors.toList()));
        ArrayList<Double> temp2 = new ArrayList(temp.stream().flatMap(x -> Arrays.stream(new Double[]{x[1]})).collect(Collectors.toList()));
        return Math.pow(variance(temp2), 0.5);
    }
    //percentchangefromyears
    public Double percentchangefromyears(HashMap<String, ArrayList<Double[]>> popdict, String code, Integer start, Integer end){
        ArrayList<Double[]> temp = new ArrayList(popdict.get(code).stream().filter(x -> x[0].equals(new Double(end)) || x[0].equals(new Double(start))).collect(Collectors.toList()));
        ArrayList<Double> temp2 = new ArrayList(temp.stream().flatMap(x -> Arrays.stream(new Double[]{x[1]})).collect(Collectors.toList()));
        return percentchange(temp2);
    } 
}
