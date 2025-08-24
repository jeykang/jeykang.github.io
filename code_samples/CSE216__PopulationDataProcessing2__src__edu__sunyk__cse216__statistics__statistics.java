/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package edu.sunyk.cse216.statistics;

import java.util.*;
import java.util.stream.*;
import edu.sunyk.cse216.statistics.MathLib;

/**
 *
 * @author techj
 */
public class statistics {
    //meanfromyears
    public Double meanfromyears(HashMap<String, ArrayList<Double[]>> popdict, String code, Integer start, Integer end){
        ArrayList<Double[]> temp = new ArrayList(popdict.get(code).stream().filter(x -> x[0] < end + 1 && x[0] >= start).collect(Collectors.toList()));
        ArrayList<Double> temp2 = new ArrayList(temp.stream().flatMap(x -> Arrays.stream(new Double[]{x[1]})).collect(Collectors.toList()));
        return MathLib.mean(temp2);
    }
    //medianfromyears
    public Double medianfromyears(HashMap<String, ArrayList<Double[]>> popdict, String code, Integer start, Integer end){
        ArrayList<Double[]> temp = new ArrayList(popdict.get(code).stream().filter(x -> x[0] < end + 1 && x[0] >= start).collect(Collectors.toList()));
        ArrayList<Double> temp2 = new ArrayList(temp.stream().flatMap(x -> Arrays.stream(new Double[]{x[1]})).collect(Collectors.toList()));
        Collections.sort(temp2);
        return MathLib.median(temp2);
    }
    //modefromyears
    public Double modefromyears(HashMap<String, ArrayList<Double[]>> popdict, String code, Integer start, Integer end){
        ArrayList<Double[]> temp = new ArrayList(popdict.get(code).stream().filter(x -> x[0] < end + 1 && x[0] >= start).collect(Collectors.toList()));
        ArrayList<Double> temp2 = new ArrayList(temp.stream().flatMap(x -> Arrays.stream(new Double[]{x[1]})).collect(Collectors.toList()));
        return MathLib.mode(temp2);
    }
    //variancefromyears
    public Double variancefromyears(HashMap<String, ArrayList<Double[]>> popdict, String code, Integer start, Integer end){
        ArrayList<Double[]> temp = new ArrayList(popdict.get(code).stream().filter(x -> x[0] < end + 1 && x[0] >= start).collect(Collectors.toList()));
        ArrayList<Double> temp2 = new ArrayList(temp.stream().flatMap(x -> Arrays.stream(new Double[]{x[1]})).collect(Collectors.toList()));
        return MathLib.variance(temp2);
    }
    //stddevfromyears
    public Double stddevfromyears(HashMap<String, ArrayList<Double[]>> popdict, String code, Integer start, Integer end){
        ArrayList<Double[]> temp = new ArrayList(popdict.get(code).stream().filter(x -> x[0] < end + 1 && x[0] >= start).collect(Collectors.toList()));
        ArrayList<Double> temp2 = new ArrayList(temp.stream().flatMap(x -> Arrays.stream(new Double[]{x[1]})).collect(Collectors.toList()));
        return MathLib.stddev(temp2);
    }
    //percentchangefromyears
    public Double percentchangefromyears(HashMap<String, ArrayList<Double[]>> popdict, String code, Integer start, Integer end){
        ArrayList<Double[]> temp = new ArrayList(popdict.get(code).stream().filter(x -> x[0].equals(new Double(end)) || x[0].equals(new Double(start))).collect(Collectors.toList()));
        ArrayList<Double> temp2 = new ArrayList(temp.stream().flatMap(x -> Arrays.stream(new Double[]{x[1]})).collect(Collectors.toList()));
        return MathLib.percentchange(temp2);
    } 
}
