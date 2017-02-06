package com.ilighti.ml;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.StringTokenizer;

import static com.ilighti.ml.CommonUtils.atof;
import static com.ilighti.ml.CommonUtils.atoi;


/**
 * <p>Describes the problem</p>
 * <p>
 * For example, if we have the following training data:
 * <pre>
 *  LABEL       ATTR1   ATTR2   ATTR3   ATTR4   ATTR5
 *  -----       -----   -----   -----   -----   -----
 *  1           0       0.1     0.2     0       0
 *  2           0       0.1     0.3    -1.2     0
 *  1           0.4     0       0       0       0
 *  2           0       0.1     0       1.4     0.5
 *  3          -0.1    -0.2     0.1     1.1     0.1
 *
 *  and bias = 1, then the components of problem are:
 *
 *  l = 5
 *  n = 6
 *
 *  y -&gt; 1 2 1 2 3
 *
 *  x -&gt; [ ] -&gt; (2,0.1) (3,0.2) (6,1) (-1,?)
 *       [ ] -&gt; (2,0.1) (3,0.3) (4,-1.2) (6,1) (-1,?)
 *       [ ] -&gt; (1,0.4) (6,1) (-1,?)
 *       [ ] -&gt; (2,0.1) (4,1.4) (5,0.5) (6,1) (-1,?)
 *       [ ] -&gt; (1,-0.1) (2,-0.2) (3,0.1) (4,1.1) (5,0.1) (6,1) (-1,?)
 * </pre>
 */
public class Problem {

    /**
     * the number of training data
     */
    public int l;

    /**
     * the number of features (including the bias feature if bias &gt;= 0)
     */
    public int n;

    /**
     * an array containing the target values
     */
    public double[] y;

    /**
     * array of sparse feature nodes
     */
    public Feature[][] x;

    /**
     * If bias &gt;= 0, we assume that one additional feature is added
     * to the end of each data instance
     */
    public double bias;

    /**
     * see {@link Problem#readProblem(File, double)}
     */
    public static Problem readFromFile(File file, double bias) throws IOException, InvalidInputDataException {
        return readProblem(file, bias);
    }

    private static Problem constructProblem(List<Double> vy, List<Feature[]> vx, int maxIndex, double bias) {
        Problem prob = new Problem();
        prob.bias = bias;
        prob.l = vy.size();
        prob.n = maxIndex;
        if (bias >= 0) {
            prob.n++;
        }
        prob.x = new Feature[prob.l][];
        for (int i = 0; i < prob.l; i++) {
            prob.x[i] = vx.get(i);

            if (bias >= 0) {
                assert prob.x[i][prob.x[i].length - 1] == null;
                prob.x[i][prob.x[i].length - 1] = new FeatureNode(maxIndex + 1, bias);
            }
        }

        prob.y = new double[prob.l];
        for (int i = 0; i < prob.l; i++)
            prob.y[i] = vy.get(i).doubleValue();

        return prob;
    }

    public static Problem readProblem(File file, double bias) throws IOException, InvalidInputDataException {
        BufferedReader fp = new BufferedReader(new FileReader(file));
        List<Double> vy = new ArrayList<Double>();
        List<Feature[]> vx = new ArrayList<Feature[]>();
        int maxIndex = 0;

        int lineNr = 0;

        try {
            while (true) {
                String line = fp.readLine();
                if (line == null) break;
                lineNr++;

                StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");
                String token;
                try {
                    token = st.nextToken();
                } catch (NoSuchElementException e) {
                    throw new InvalidInputDataException("empty line", file, lineNr, e);
                }

                try {
                    vy.add(atof(token));
                } catch (NumberFormatException e) {
                    throw new InvalidInputDataException("invalid label: " + token, file, lineNr, e);
                }

                int m = st.countTokens() / 2;
                Feature[] x;
                int j = -1;
                if (bias >= 0) {
                    x = new Feature[m + 1];
                    j++;
                    x[j] = new FeatureNode(0, bias);
                } else {
                    x = new Feature[m];
                }
                int indexBefore = 0;
                while(st.hasMoreTokens()) {
                    j++;
                    token = st.nextToken();
                    int index;
                    try {
                        index = atoi(token);
                    } catch (NumberFormatException e) {
                        throw new InvalidInputDataException("invalid index: " + token, file, lineNr, e);
                    }

                    // assert that indices are valid and sorted
                    if (index < 0) throw new InvalidInputDataException("invalid index: " + index, file, lineNr);
                    if (index <= indexBefore)
                        throw new InvalidInputDataException("indices must be sorted in ascending order", file, lineNr);
                    indexBefore = index;

                    token = st.nextToken();
                    try {
                        double value = atof(token);
                        x[j] = new FeatureNode(index, value);
                    } catch (NumberFormatException e) {
                        throw new InvalidInputDataException("invalid value: " + token, file, lineNr);
                    }
                }
                if (m > 0) {
                    maxIndex = Math.max(maxIndex, x[m - 1].getIndex());
                }

                vx.add(x);
            }

            return constructProblem(vy, vx, maxIndex, bias);
        } finally {
            fp.close();
        }
    }
}
