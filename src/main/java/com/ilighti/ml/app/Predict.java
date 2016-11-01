package com.ilighti.ml.app;
;
import com.ilighti.ml.InvalidInputDataException;
import com.ilighti.ml.Problem;
import com.ilighti.ml.ftrl.Ftrl;
import com.ilighti.ml.ftrl.Model;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;


/**
 * Created by rain on 16-6-14.
 */
public class Predict {
    private String testFilename;
    private String modelFilename;
    private String outputFilename;

    public void run(String[] args) {
        parseCommandLine(args);
        try {
            Problem problem = Problem.readProblem(new File(testFilename), -1.);
            Model model = Model.load(new File(modelFilename));
            if(model == null) {
                exitWithHelp();
            }
            BufferedWriter bufferedWriter = new BufferedWriter(new FileWriter(new File(outputFilename)));
            bufferedWriter.write("labels " + join(model.getLabel()) + "\n");
            Ftrl ftrl = new Ftrl();
            double[] probabilities = new double[model.getNrClass()];
            int totalCorrect = 0;
            for(int i=0; i<problem.l; i++) {
                Double label = ftrl.predict(problem.x[i], model, probabilities);
                if(label.equals(problem.y[i])) {
                    ++totalCorrect;
                }
                bufferedWriter.write(label.toString() + " " + join(probabilities) + "\n");
            }
            bufferedWriter.close();
            System.out.printf("correct: %d%n", totalCorrect);
            System.out.printf("Cross Validation Accuracy = %g%%%n", 100.0 * totalCorrect / problem.l);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InvalidInputDataException e) {
            e.printStackTrace();
        }
    }
    private String join(int[] integers) {
        StringBuilder stringBuilder = new StringBuilder();
        for (int i : integers) {
            stringBuilder.append(i).append(" ");
        }
        return stringBuilder.toString();
    }
    private String join(double[] reals) {
        StringBuilder stringBuilder = new StringBuilder();
        for (double d : reals) {
            stringBuilder.append(d).append(" ");
        }
        return stringBuilder.toString();
    }
    public void parseCommandLine(String argv[]) {
        int i = 0;
        if(argv.length < 3) {
            exitWithHelp();
        }

        testFilename = argv[i];
        modelFilename = argv[i+1];
        outputFilename = argv[i+2];
    }

    private static void exitWithHelp() {
        System.out.printf("Usage: predict test_file model_file output_file%n");
        System.exit(1);
    }
    public static void main(String[] args) {
        new Predict().run(args);
    }
}
