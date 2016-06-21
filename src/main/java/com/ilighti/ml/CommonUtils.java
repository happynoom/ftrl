package com.ilighti.ml;

import java.io.PrintStream;

/**
 * Created by rain on 16-6-14.
 */
public class CommonUtils {
    private static Object OUTPUT_MUTEX = new Object();
    private static PrintStream DEBUG_OUTPUT = System.out;

    public static void info(String message) {
        synchronized (OUTPUT_MUTEX) {
            if (DEBUG_OUTPUT == null) return;
            DEBUG_OUTPUT.printf(message);
            DEBUG_OUTPUT.flush();
        }
    }

    public static void info(String format, Object... args) {
        synchronized (OUTPUT_MUTEX) {
            if (DEBUG_OUTPUT == null) return;
            DEBUG_OUTPUT.printf(format, args);
            DEBUG_OUTPUT.flush();
        }
    }

    /**
     * @param s the string to parse for the double value
     * @throws IllegalArgumentException if s is empty or represents NaN or Infinity
     * @throws NumberFormatException    see {@link Double#parseDouble(String)}
     */
    public static double atof(String s) {
        if (s == null || s.length() < 1) throw new IllegalArgumentException("Can't convert empty string to integer");
        double d = Double.parseDouble(s);
        if (Double.isNaN(d) || Double.isInfinite(d)) {
            throw new IllegalArgumentException("NaN or Infinity in input: " + s);
        }
        return (d);
    }

    /**
     * @param s the string to parse for the integer value
     * @throws IllegalArgumentException if s is empty
     * @throws NumberFormatException    see {@link Integer#parseInt(String)}
     */
    public static int atoi(String s) throws NumberFormatException {
        if (s == null || s.length() < 1) throw new IllegalArgumentException("Can't convert empty string to integer");
        // Integer.parseInt doesn't accept '+' prefixed strings
        if (s.charAt(0) == '+') s = s.substring(1);
        return Integer.parseInt(s);
    }

    /**
     * Java5 'backport' of Arrays.copyOf
     */
    public static double[] copyOf(double[] original, int newLength) {
        double[] copy = new double[newLength];
        System.arraycopy(original, 0, copy, 0, Math.min(original.length, newLength));
        return copy;
    }

    /**
     * Java5 'backport' of Arrays.copyOf
     */
    public static int[] copyOf(int[] original, int newLength) {
        int[] copy = new int[newLength];
        System.arraycopy(original, 0, copy, 0, Math.min(original.length, newLength));
        return copy;
    }

    public static void swap(double[] array, int idxA, int idxB) {
        double temp = array[idxA];
        array[idxA] = array[idxB];
        array[idxB] = temp;
    }

    public static void swap(int[] array, int idxA, int idxB) {
        int temp = array[idxA];
        array[idxA] = array[idxB];
        array[idxB] = temp;
    }
}
