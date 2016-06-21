package com.ilighti.ml.ftrl;

import java.io.*;
import com.google.gson.*;
/**
 * Created by rain on 16-6-14.
 */
public class Model implements Serializable {

    double bias;

    /**
     * label of each class
     */
    int[] label;

    int nrClass;

    int nrFeature;

    FtrlSolver[] ftrlSolvers;

    public void save(File file) {
        try {
            BufferedWriter bw = new BufferedWriter(new FileWriter(file));
            bw.write(new Gson().toJson(this));
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static Model load(File file) {
        try {
            BufferedReader br = new BufferedReader(new FileReader(file));
            Model model = new Gson().fromJson(br.readLine(), Model.class);
            br.close();
            return model;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public int[] getLabel() {
        return label;
    }

    public void setLabel(int[] label) {
        this.label = label;
    }

    public int getNrClass() {
        return nrClass;
    }

    public void setNrClass(int nrClass) {
        this.nrClass = nrClass;
    }

    public int getNrFeature() {
        return nrFeature;
    }

    public void setNrFeature(int nrFeature) {
        this.nrFeature = nrFeature;
    }

    public FtrlSolver[] getFtrlSolvers() {
        return ftrlSolvers;
    }

    public void setFtrlSolvers(FtrlSolver[] ftrlSolvers) {
        this.ftrlSolvers = ftrlSolvers;
    }
}
