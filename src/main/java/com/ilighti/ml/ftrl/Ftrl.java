package com.ilighti.ml.ftrl;

import com.ilighti.ml.Feature;
import com.ilighti.ml.Problem;

import java.util.Random;

import static com.ilighti.ml.CommonUtils.copyOf;
import static com.ilighti.ml.CommonUtils.swap;

/**
 * Created by rain on 16-6-14.
 */
public class Ftrl {
    private static final long DEFAULT_RANDOM_SEED = 0L;
    static Random random = new Random(DEFAULT_RANDOM_SEED);
    private static int MAX_ITER = 1;
    /**
     * used as complex return type
     */
    private static class GroupClassesReturn {

        final int[] count;
        final int[] label;
        final int nrClass;
        final int[] start;

        GroupClassesReturn(int nrClass, int[] label, int[] start, int[] count) {
            this.nrClass = nrClass;
            this.label = label;
            this.start = start;
            this.count = count;
        }
    }

    private static GroupClassesReturn groupClasses(Problem prob, int[] perm) {
        int l = prob.l;
        int maxNrClass = 16;
        int nrClass = 0;

        int[] label = new int[maxNrClass];
        int[] count = new int[maxNrClass];
        int[] dataLabel = new int[l];
        int i;

        for (i = 0; i < l; i++) {
            int thisLabel = (int) prob.y[i];
            int j;
            for (j = 0; j < nrClass; j++) {
                if (thisLabel == label[j]) {
                    ++count[j];
                    break;
                }
            }
            dataLabel[i] = j;
            if (j == nrClass) {
                if (nrClass == maxNrClass) {
                    maxNrClass *= 2;
                    label = copyOf(label, maxNrClass);
                    count = copyOf(count, maxNrClass);
                }
                label[nrClass] = thisLabel;
                count[nrClass] = 1;
                ++nrClass;
            }
        }

        //
        // Labels are ordered by their first occurrence in the training set.
        // However, for two-class sets with -1/+1 labels and -1 appears first,
        // we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
        //
        if (nrClass == 2 && label[0] == -1 && label[1] == 1) {
            swap(label, 0, 1);
            swap(count, 0, 1);
            for (i = 0; i < l; i++) {
                if (dataLabel[i] == 0)
                    dataLabel[i] = 1;
                else
                    dataLabel[i] = 0;
            }
        }

        int[] start = new int[nrClass];
        start[0] = 0;
        for (i = 1; i < nrClass; i++)
            start[i] = start[i - 1] + count[i - 1];
        for (i = 0; i < l; i++) {
            perm[start[dataLabel[i]]] = i;
            ++start[dataLabel[i]];
        }
        start[0] = 0;
        for (i = 1; i < nrClass; i++)
            start[i] = start[i - 1] + count[i - 1];

        return new GroupClassesReturn(nrClass, label, start, count);
    }

    public Model train(Problem prob, Parameter parameter) {
        if (prob == null) throw new IllegalArgumentException("problem must not be null");
        if (prob.n == 0) throw new IllegalArgumentException("problem has zero features");
        if (prob.l == 0) throw new IllegalArgumentException("problem has zero instances");
        for (Feature[] nodes : prob.x) {
            int indexBefore = 0;
            for (Feature n : nodes) {
                if (n.getIndex() <= indexBefore) {
                    throw new IllegalArgumentException("feature nodes must be sorted by index in ascending order");
                }
                indexBefore = n.getIndex();
            }
        }
        int l = prob.l;
        int n = prob.n;
        int[] perm = new int[l];
        Model model = new Model();

        if (prob.bias >= 0)
            model.nrFeature = n - 1;
        else
            model.nrFeature = n;

        model.bias = prob.bias;
        GroupClassesReturn rv = groupClasses(prob, perm);
        int nrClass = rv.nrClass;
        int[] label = rv.label;
        int[] start = rv.start;
        int[] count = rv.count;
        checkProblemSize(n, nrClass);

        model.nrClass = nrClass;
        model.label = new int[nrClass];
        for (int i = 0; i < nrClass; i++)
            model.label[i] = label[i];

        // constructing the subproblem
        Feature[][] x = new Feature[l][];
        for (int i = 0; i < l; i++)
            x[i] = prob.x[perm[i]];

        Problem subProb = new Problem();
        subProb.l = l;
        subProb.n = n;
        subProb.x = new Feature[subProb.l][];
        subProb.y = new double[subProb.l];
        for (int k = 0; k < subProb.l; k++)
            subProb.x[k] = x[k];

        if (nrClass == 2) {
            model.ftrlSolvers = new FtrlSolver[1];
            model.ftrlSolvers[0] = new FtrlSolver(parameter.alpha, parameter.beta, parameter.lambdaOne, parameter.lambdaTwo);
            int e0 = start[0] + count[0];
            int k = 0;
            for (; k < e0; k++)
                subProb.y[k] = +1;
            for (; k < subProb.l; k++)
                subProb.y[k] = 0;
            //random problem
            int[] rand = new int[subProb.l];
            for (k = 0; k < subProb.l; k++) {
                rand[k] = k;
            }
            //Random random = new Random();
            for (k = 0; k < subProb.l; k++) {
                int val = random.nextInt(l - k);
                swap(rand, k, k + val);
            }

            for(int iter = 0; iter < MAX_ITER; iter ++) {
                for (int i = 0; i < subProb.l; i++) {
                    model.ftrlSolvers[0].trainOne(subProb.x[rand[i]], subProb.y[rand[i]]);
                }
            }
        } else {
            model.ftrlSolvers = new FtrlSolver[nrClass];
            for (int i = 0; i < nrClass; i++) {
                model.ftrlSolvers[i] = new FtrlSolver(parameter.alpha, parameter.beta, parameter.lambdaOne, parameter.lambdaTwo);
                int si = start[i];
                int ei = si + count[i];

                int k = 0;
                for (; k < si; k++)
                    subProb.y[k] = 0;
                for (; k < ei; k++)
                    subProb.y[k] = +1;
                for (; k < subProb.l; k++)
                    subProb.y[k] = 0;
                //random problem
                int[] rand = new int[subProb.l];
                for (k = 0; k < subProb.l; k++) {
                    rand[k] = k;
                }
                //Random random = new Random();
                for (k = 0; k < subProb.l; k++) {
                    int val = random.nextInt(l - k);
                    swap(rand, k, k + val);
                }
                for(int iter = 0; iter < MAX_ITER; iter ++) {
                    for (int j = 0; j < subProb.l; j++) {
                        model.ftrlSolvers[i].trainOne(subProb.x[rand[j]], subProb.y[rand[j]]);
                    }
                }
            }
        }
        return model;
    }

//    private <S, T> T getOrDefault(Map<S, T> data, S key, T defaultValue) {
//        T val = data.get(key);
//        if(val == null) {
//            return defaultValue;
//        }
//        return val;
//    }

    public double predict(Feature[] x, Model model, double[] probabilities) {
        double label = -1;
        double maxProbability = 0.;

        if (model.nrClass == 2) {
            probabilities[0] = model.ftrlSolvers[0].predictOne(x);
            probabilities[1] = 1. - probabilities[0];
            label = probabilities[0] > probabilities[1] ? model.label[0] : model.label[1];
        } else {
            for (int i = 0; i < model.nrClass; i++) {
                probabilities[i] = model.ftrlSolvers[i].predictOne(x);
                if (maxProbability <= probabilities[i]) {
                    maxProbability = probabilities[i];
                    label = model.label[i];
                }
            }
            double sum = 0.;
            for (int i = 0; i < model.nrClass; i++) {
                sum += probabilities[i];
            }
            for (int i = 0; i < model.nrClass; i++) {
                probabilities[i] = probabilities[i] / sum;
            }
        }
        return label;
    }

    /**
     * verify the size and throw an exception early if the problem is too large
     */
    private static void checkProblemSize(int n, int nrClass) {
        if (n >= Integer.MAX_VALUE / nrClass || n * nrClass < 0) {
            throw new IllegalArgumentException("'number of classes' * 'number of instances' is too large: " + nrClass + "*" + n);
        }
    }

    public void crossValidation(Problem prob, Parameter param, int nrFold, double[] target) {
        int i;
        int l = prob.l;
        int[] perm = new int[l];

        if (nrFold > l) {
            nrFold = l;
            System.err.println("WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)");
        }
        int[] foldStart = new int[nrFold + 1];

        for (i = 0; i < l; i++)
            perm[i] = i;
        for (i = 0; i < l; i++) {
            int j = i + random.nextInt(l - i);
            swap(perm, i, j);
        }
        for (i = 0; i <= nrFold; i++)
            foldStart[i] = i * l / nrFold;

        for (i = 0; i < nrFold; i++) {
            int begin = foldStart[i];
            int end = foldStart[i + 1];
            int j, k;
            Problem subProb = new Problem();

            subProb.bias = prob.bias;
            subProb.n = prob.n;
            subProb.l = l - (end - begin);
            subProb.x = new Feature[subProb.l][];
            subProb.y = new double[subProb.l];

            k = 0;
            for (j = 0; j < begin; j++) {
                subProb.x[k] = prob.x[perm[j]];
                subProb.y[k] = prob.y[perm[j]];
                ++k;
            }
            for (j = end; j < l; j++) {
                subProb.x[k] = prob.x[perm[j]];
                subProb.y[k] = prob.y[perm[j]];
                ++k;
            }
            Model submodel = train(subProb, param);
            double[] probabilities = new double[submodel.getNrClass()];
            for (j = begin; j < end; j++) {
                target[perm[j]] = predict(prob.x[perm[j]], submodel, probabilities);
            }
        }
    }
}
