package com.ilighti.ml.ftrl;

import com.ilighti.ml.Feature;
import com.ilighti.ml.Problem;

import java.util.Map;
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
        final int nr_class;
        final int[] start;

        GroupClassesReturn(int nr_class, int[] label, int[] start, int[] count) {
            this.nr_class = nr_class;
            this.label = label;
            this.start = start;
            this.count = count;
        }
    }

    private static GroupClassesReturn groupClasses(Problem prob, int[] perm) {
        int l = prob.l;
        int max_nr_class = 16;
        int nr_class = 0;

        int[] label = new int[max_nr_class];
        int[] count = new int[max_nr_class];
        int[] data_label = new int[l];
        int i;

        for (i = 0; i < l; i++) {
            int this_label = (int) prob.y[i];
            int j;
            for (j = 0; j < nr_class; j++) {
                if (this_label == label[j]) {
                    ++count[j];
                    break;
                }
            }
            data_label[i] = j;
            if (j == nr_class) {
                if (nr_class == max_nr_class) {
                    max_nr_class *= 2;
                    label = copyOf(label, max_nr_class);
                    count = copyOf(count, max_nr_class);
                }
                label[nr_class] = this_label;
                count[nr_class] = 1;
                ++nr_class;
            }
        }

        //
        // Labels are ordered by their first occurrence in the training set.
        // However, for two-class sets with -1/+1 labels and -1 appears first,
        // we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
        //
        if (nr_class == 2 && label[0] == -1 && label[1] == 1) {
            swap(label, 0, 1);
            swap(count, 0, 1);
            for (i = 0; i < l; i++) {
                if (data_label[i] == 0)
                    data_label[i] = 1;
                else
                    data_label[i] = 0;
            }
        }

        int[] start = new int[nr_class];
        start[0] = 0;
        for (i = 1; i < nr_class; i++)
            start[i] = start[i - 1] + count[i - 1];
        for (i = 0; i < l; i++) {
            perm[start[data_label[i]]] = i;
            ++start[data_label[i]];
        }
        start[0] = 0;
        for (i = 1; i < nr_class; i++)
            start[i] = start[i - 1] + count[i - 1];

        return new GroupClassesReturn(nr_class, label, start, count);
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
        int nr_class = rv.nr_class;
        int[] label = rv.label;
        int[] start = rv.start;
        int[] count = rv.count;
        checkProblemSize(n, nr_class);

        model.nrClass = nr_class;
        model.label = new int[nr_class];
        for (int i = 0; i < nr_class; i++)
            model.label[i] = label[i];

        // constructing the subproblem
        Feature[][] x = new Feature[l][];
        for (int i = 0; i < l; i++)
            x[i] = prob.x[perm[i]];

        Problem sub_prob = new Problem();
        sub_prob.l = l;
        sub_prob.n = n;
        sub_prob.x = new Feature[sub_prob.l][];
        sub_prob.y = new double[sub_prob.l];
        for (int k = 0; k < sub_prob.l; k++)
            sub_prob.x[k] = x[k];

        if (nr_class == 2) {
            model.ftrlSolvers = new FtrlSolver[1];
            model.ftrlSolvers[0] = new FtrlSolver(parameter.alpha, parameter.beta, parameter.lambdaOne, parameter.lambdaTwo);
            int e0 = start[0] + count[0];
            int k = 0;
            for (; k < e0; k++)
                sub_prob.y[k] = +1;
            for (; k < sub_prob.l; k++)
                sub_prob.y[k] = 0;

            for(int iter = 0; iter < MAX_ITER; iter ++) {
                for (int i = 0; i < sub_prob.l; i++) {
                    for(int r = 0; r < getOrDefault(parameter.labelWeigths, (int) prob.y[perm[i]], 1); r++) {
                        model.ftrlSolvers[0].trainOne(sub_prob.x[i], sub_prob.y[i]);
                    }
                }
            }
        } else {
            model.ftrlSolvers = new FtrlSolver[nr_class];
            for (int i = 0; i < nr_class; i++) {
                model.ftrlSolvers[i] = new FtrlSolver(parameter.alpha, parameter.beta, parameter.lambdaOne, parameter.lambdaTwo);
                int si = start[i];
                int ei = si + count[i];

                int k = 0;
                for (; k < si; k++)
                    sub_prob.y[k] = 0;
                for (; k < ei; k++)
                    sub_prob.y[k] = +1;
                for (; k < sub_prob.l; k++)
                    sub_prob.y[k] = 0;
                for(int iter = 0; iter < MAX_ITER; iter ++) {
                    for (int j = 0; j < sub_prob.l; j++) {
                        for(int r = 0; r < getOrDefault(parameter.labelWeigths, (int) prob.y[perm[j]], 1); r++) {
                            model.ftrlSolvers[i].trainOne(sub_prob.x[j], sub_prob.y[j]);
                        }
                    }
                }
            }
        }
        return model;
    }

    private <S, T> T getOrDefault(Map<S, T> data, S key, T defaultValue) {
        T val = data.get(key);
        if(val == null) {
            return defaultValue;
        }
        return val;
    }

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
    private static void checkProblemSize(int n, int nr_class) {
        if (n >= Integer.MAX_VALUE / nr_class || n * nr_class < 0) {
            throw new IllegalArgumentException("'number of classes' * 'number of instances' is too large: " + nr_class + "*" + n);
        }
    }

    public void crossValidation(Problem prob, Parameter param, int nr_fold, double[] target) {
        int i;
        int l = prob.l;
        int[] perm = new int[l];

        if (nr_fold > l) {
            nr_fold = l;
            System.err.println("WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)");
        }
        int[] fold_start = new int[nr_fold + 1];

        for (i = 0; i < l; i++)
            perm[i] = i;
        for (i = 0; i < l; i++) {
            int j = i + random.nextInt(l - i);
            swap(perm, i, j);
        }
        for (i = 0; i <= nr_fold; i++)
            fold_start[i] = i * l / nr_fold;

        for (i = 0; i < nr_fold; i++) {
            int begin = fold_start[i];
            int end = fold_start[i + 1];
            int j, k;
            Problem subprob = new Problem();

            subprob.bias = prob.bias;
            subprob.n = prob.n;
            subprob.l = l - (end - begin);
            subprob.x = new Feature[subprob.l][];
            subprob.y = new double[subprob.l];

            k = 0;
            for (j = 0; j < begin; j++) {
                subprob.x[k] = prob.x[perm[j]];
                subprob.y[k] = prob.y[perm[j]];
                ++k;
            }
            for (j = end; j < l; j++) {
                subprob.x[k] = prob.x[perm[j]];
                subprob.y[k] = prob.y[perm[j]];
                ++k;
            }
            Model submodel = train(subprob, param);
            double[] probabilities = new double[submodel.getNrClass()];
            for (j = begin; j < end; j++) {
                target[perm[j]] = predict(prob.x[perm[j]], submodel, probabilities);
            }
        }
    }
}
