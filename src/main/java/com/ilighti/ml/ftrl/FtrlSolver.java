package com.ilighti.ml.ftrl;

import com.ilighti.ml.Feature;

import java.io.Serializable;
import java.util.HashMap;

/**
 * Created by rain on 16-6-14.
 */
public class FtrlSolver implements Serializable {
    private double alpha = .1;
    private double beta = 1.0;
    private double lambdaOne = .1;
    private double lambdaTwo = 1.;
    private HashMap<Integer, Double> omega = null;
    private HashMap<Integer, Double> zed = null;
    private HashMap<Integer, Double> fieldSum = null;

    public FtrlSolver() {
        omega = new HashMap<Integer, Double>();
        zed = new HashMap<Integer, Double>();
        fieldSum = new HashMap<Integer, Double>();
    }

    public FtrlSolver(double alpha, double beta, double lambdaOne, double lambdaTwo) {
        this.alpha = alpha;
        this.beta = beta;
        this.lambdaOne = lambdaOne;
        this.lambdaTwo = lambdaTwo;
        omega = new HashMap<Integer, Double>();
        zed = new HashMap<Integer, Double>();
        fieldSum = new HashMap<Integer, Double>();
    }

    public void trainOne(Feature[] x, Double y) {
        for (Feature node : x) {
            Integer index = node.getIndex();
            Double z = zed.get(index);
            Double ni = fieldSum.get(index);
            Double weight;
            if (z == null)
                z = 0.0;
            if (ni == null)
                ni = 0.0;
            if (Math.abs(z) > lambdaOne) {
                int sgn = z > 0 ? 1 : -1;
                weight = -(z - sgn * lambdaOne) / ((beta + Math.sqrt(ni)) / alpha + lambdaTwo);
                omega.put(index, weight);
            } else {
                omega.put(index, 0.);
            }
        }
        double p = sigmoid(x, omega);
        for (Feature node : x) {
            Integer index = node.getIndex();
            Double z = zed.get(index);
            Double ni = fieldSum.get(index);
            Double weight = omega.get(index);
            if (z == null)
                z = 0.0;
            if (ni == null)
                ni = 0.0;
            if (weight == null)
                weight = 0.0;
            Double gi = (p - y) * node.getValue();
            Double sigma = (Math.sqrt(ni + gi * gi) - Math.sqrt(ni)) / alpha;
            z = z + gi - sigma * weight;
            ni = ni + gi * gi;
            if (weight.compareTo(0.0) != 0) {
                omega.put(index, weight);
            }
            if (z.compareTo(0.0) != 0) {
                zed.put(index, z);
            }
            if (ni.compareTo(0.0) != 0) {
                fieldSum.put(index, ni);
            }
        }
    }

    public double predictOne(Feature[] x) {
        return sigmoid(x, omega);
    }

    private double sigmoid(Feature[] x, HashMap<Integer, Double> omega) {
        double ret = 0.0;
        for (Feature node : x) {
            Double weight = omega.get(node.getIndex());
            if (weight == null)
                weight = 0.0;
            ret += weight * node.getValue();
        }
        return 1.0 / (1.0 + Math.exp(-ret));
    }

    public double getAlpha() {
        return alpha;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
    }

    public double getBeta() {
        return beta;
    }

    public void setBeta(double beta) {
        this.beta = beta;
    }

    public double getLambdaOne() {
        return lambdaOne;
    }

    public void setLambdaOne(double lambdaOne) {
        this.lambdaOne = lambdaOne;
    }

    public double getLambdaTwo() {
        return lambdaTwo;
    }

    public void setLambdaTwo(double lambdaTwo) {
        this.lambdaTwo = lambdaTwo;
    }

    public HashMap<Integer, Double> getOmega() {
        return omega;
    }

    public void setOmega(HashMap<Integer, Double> omega) {
        this.omega = omega;
    }

    public HashMap<Integer, Double> getZed() {
        return zed;
    }

    public void setZed(HashMap<Integer, Double> zed) {
        this.zed = zed;
    }

    public HashMap<Integer, Double> getFieldSum() {
        return fieldSum;
    }

    public void setFieldSum(HashMap<Integer, Double> fieldSum) {
        this.fieldSum = fieldSum;
    }
}
