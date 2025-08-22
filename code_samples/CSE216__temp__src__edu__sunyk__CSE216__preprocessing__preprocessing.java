/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package edu.sunyk.CSE216.preprocessing;

import java.util.*;
import com.opencsv.CSVReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/**
 *
 * @author gch04
 */
public class preprocessing {

    CSVReader csv_file;
    List<String[]> csv_reader;
    List<String> year;

    public preprocessing(String popPath) throws FileNotFoundException, IOException {
        csv_file = new CSVReader(new FileReader(popPath));
        csv_reader = csv_file.readAll();
        int line_count = 0;
        for(String[] row: csv_reader){
            if(line_count == 0){
                year = Arrays.asList(row).subList(2, row.length);
                break;
            }
        }

    }
    
    public void ye(){
        for(int x = 0; x < year.size(); x++){
            String yearColumn = year.get(x);
            System.out.println((x + 1) + ". " + yearColumn);
        }
    }
    
    public ArrayList<String> year_column(){
        ArrayList<String> ear = new ArrayList();
        for(int i = 0; i < year.size(); i++){
            ear.add(year.get(i));
        }
        return ear;
    }
    
    public HashMap<String, ArrayList<Double[]>> dict(){
        HashMap<String, ArrayList<Double[]>> dic = new HashMap();
        int count_line = 0;
        String[] ear = new String[2];
        for(String[] row: csv_reader){
            if(count_line == 0){
                ear = row;
            }
            else{
                dic.put(row[1], new ArrayList<Double[]>());
                for(int i = 0; i < row.length - 2; i++){
                    if(row[i + 2].equals("")){
                        dic.get(row[1]).add(new Double[]{Double.parseDouble(ear[i + 2]), 0.0});
                    }
                    else{
                        dic.get(row[1]).add(new Double[]{Double.parseDouble(ear[i + 2]), Double.parseDouble(row[i+2])});
                    }
                }
            }
            count_line++;
        }
        return dic;
    }
    
    public ArrayList<String> countries(){
        ArrayList<String> country = new ArrayList();
        int line_count = 0;
        for(String[] row: csv_reader){
            if(line_count > 0){
                country.add(row[0]);
            }
            line_count++;
        }
        return country;
    }
    
    public void num_country(){
        ArrayList<String> country = countries();
        for(int i = 0; i < country.size(); i++){
            String countryNum = country.get(i);
            System.out.println((i + 1) + ". " + countryNum);
        }
    }
    
    public String country_list(int num){
        ArrayList<String> country = new ArrayList();
        int line_count = 0;
        for(String[] row: csv_reader){
            if(line_count > 0){
                country.add(row[0]);
            }
            line_count++;
        }
        return country.get(num);
    }
    
    public static void main(String[] args) throws IOException {
        preprocessing test = new preprocessing("YearlyPopulation.csv");
        HashMap testing = test.dict();
        Iterator iterator = testing.keySet().iterator();

        while (iterator.hasNext()) {
            String key = iterator.next().toString();
            String value = testing.get(key).toString();

            System.out.println(key + " " + value);
        }
    }
}