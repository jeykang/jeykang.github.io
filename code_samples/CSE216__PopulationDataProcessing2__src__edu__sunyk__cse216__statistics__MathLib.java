package edu.sunyk.cse216.statistics;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 *
 * @author SUNY Korea CS
 */

@FunctionalInterface
interface StatsUtil<T,R> {
 R calculateStats(T l);
} 
public class MathLib {
    //Calculate mean
    public static double mean(List<Double> someList){
        StatsUtil<List<Double>, Double> meancalculator = l -> l.stream().collect(Collectors.averagingDouble(x -> x));
        return meancalculator.calculateStats(someList); 

    }
    
    

    //Calculate variance
    public static double variance(List<Double> someList){
        StatsUtil<List<Double>, Double> variancecalc = l -> {
            ArrayList<Double> tempsum = new ArrayList();
            l.forEach(x -> tempsum.add(Math.pow((x - mean(l)), 2)));
            return tempsum.stream().reduce((x, y) -> x + y).orElse(-1.0) / (l.size()-1);
        };
        return variancecalc.calculateStats(someList);
    }
    
    //Calculate standard deviation
    //Taken from: https://stackoverflow.com/questions/1735870/simple-statistics-java-packages-for-calculating-mean-standard-deviation-etc
    public static double stddev(List<Double> someList){
        StatsUtil<List<Double>, Double> stddevcalc = l -> Math.sqrt(MathLib.variance(l));
        return stddevcalc.calculateStats(someList);
    }
    
    //Calculate median: code taken from https://codippa.com/how-to-calculate-median-from-array-values-in-java/
    public static double median(List<Double> someList) {
        StatsUtil<List<Double>, Double> mediancalc = l -> l.stream().filter(x -> x == l.get(l.size()/2 - 1) || x == l.get(l.size()/2)).reduce((x, y) -> x+y).orElse(0.0) / 2;
        return mediancalc.calculateStats(someList); 
    }
    
    //Calculate mode. Code taken from https://www.sanfoundry.com/java-program-find-mode-data-set/
    public static double mode(List<Double> someList) 
    {
        StatsUtil<List<Double>, Double> modecalc = l -> {
            final ArrayList<Double> sortedCount = new ArrayList();
            l.forEach(x -> sortedCount.add(x));
            ArrayList<Double> newCount = new ArrayList(sortedCount.stream().sorted().collect(Collectors.toList()));
            return l.stream().filter(x -> Collections.frequency(l, x) == newCount.get(newCount.size()-1)).collect(Collectors.toList()).get(0);
        };
        return modecalc.calculateStats(someList); 
    }
    
    public static double percentchange(List<Double> someList){
        StatsUtil<List<Double>, Double> pchangecalc = l -> l.stream().reduce((x, y) -> (y - x)/x).orElse(-1.0) * 100;
        return pchangecalc.calculateStats(someList);
    }
}
