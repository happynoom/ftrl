package com.ilighti.ml.app;

import com.ilighti.ml.CommonUtils;
import com.ilighti.ml.InvalidInputDataException;
import com.ilighti.ml.Problem;
import com.ilighti.ml.ftrl.Ftrl;
import com.ilighti.ml.ftrl.Model;
import com.ilighti.ml.ftrl.Parameter;

import java.io.File;
import java.io.IOException;

/**
 * Created by rain on 16-6-14.
 */
public class Train {
    private String inputFilename;
    private String modelFilename;
    Parameter param = new Parameter();
    Model model;
    private int nr_fold = 0;

    public void do_cross_validation(Problem prob, int nr_fold) {
        double total_error = 0;
        double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
        double[] target = new double[prob.l];

        long start, stop;
        start = System.currentTimeMillis();
        new Ftrl().crossValidation(prob, param, nr_fold, target);
        stop = System.currentTimeMillis();
        System.out.println("time: " + (stop - start) + " ms");

        int total_correct = 0;
        for (int i = 0; i < prob.l; i++)
            if (target[i] == prob.y[i]) ++total_correct;

        System.out.printf("correct: %d%n", total_correct);
        System.out.printf("Cross Validation Accuracy = %g%%%n", 100.0 * total_correct / prob.l);
    }

    public void run(String[] args) {
        parse_command_line(args);
        try {
            Problem prob = Problem.readFromFile(new File(inputFilename), param.getBias());
            if (nr_fold > 0) {
                do_cross_validation(prob, nr_fold);
            } else {
                model = new Ftrl().train(prob, param);
                model.save(new File(modelFilename));
            }
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InvalidInputDataException e) {
            e.printStackTrace();
        }
    }

    private void exit_with_help() {
        System.out.printf("Usage: train [options] training_set_file [model_file]%n" //
                + "options:%n"
                + "-a alpha : set the parameter alpha (default 0.1)%n"
                + "-b beta : set the beta (default 1)%n"
                + "-o lambda 1 : set the lambda 1 (default 1)%n"
                + "-t lambda 2 : set the lambda 2 (default 1)%n"
                + "-s bias : set the bias (default -1)%n"
                + "-v nr_fold : do nr_fold validation"
                + "-w${label} weight : weight of the samples with ${label}");
        System.exit(1);
    }

    public void parse_command_line(String argv[]) {
        int i;
        // parse options
        for (i = 0; i < argv.length; i++) {
            if (argv[i].charAt(0) != '-') break;
            if (++i >= argv.length) exit_with_help();
            switch (argv[i - 1].charAt(1)) {
                case 'a':
                    param.setAlpha(CommonUtils.atof(argv[i]));
                    break;
                case 'b':
                    param.setBeta(CommonUtils.atof(argv[i]));
                    break;
                case 'o':
                    param.setLambdaOne(CommonUtils.atof(argv[i]));
                    break;
                case 't':
                    param.setLambdaTwo(CommonUtils.atof(argv[i]));
                    break;
                case 's':
                    param.setBias(CommonUtils.atof(argv[i]));
                    break;
                case 'v':
                    nr_fold = CommonUtils.atoi(argv[i]);
                    break;
                case 'w':
                    int weightLabel = Integer.parseInt(argv[i - 1].substring(2));
                    int weight = Integer.parseInt(argv[i]);
                    param.labelWeigths.put(weightLabel,weight);
                    break;
                default:
                    System.err.println("unknown option");
                    exit_with_help();
            }
        }

        // determine filenames

        if (i >= argv.length) exit_with_help();

        inputFilename = argv[i];

        if (i < argv.length - 1)
            modelFilename = argv[i + 1];
        else {
            int p = argv[i].lastIndexOf('/');
            ++p; // whew...
            modelFilename = argv[i].substring(p) + ".model";
        }
    }

    public static void main(String[] args) {
        new Train().run(args);
    }
}
