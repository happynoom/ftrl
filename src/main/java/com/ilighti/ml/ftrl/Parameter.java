package com.ilighti.ml.ftrl;

/**
 * Created by rain on 16-6-14.
 */
public class Parameter {
    public double alpha = 0.1;
    public double beta = 1.;
    public double lambdaOne = 1.;
    public double lambdaTwo = 1.;
    public double bias = 1.;

    public Parameter() {
    }

    public Parameter(double alpha, double beta, double lambdaOne, double lambdaTwo, double bias) {
        this.alpha = alpha;
        this.beta = beta;
        this.lambdaOne = lambdaOne;
        this.lambdaTwo = lambdaTwo;
        this.bias = bias;
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

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }
}
