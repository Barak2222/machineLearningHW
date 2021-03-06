package HomeWork5;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.text.MessageFormat;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instance;
import weka.core.Instances;

public class MainHW5 {
	private static final int ALPHA = 1;
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	
	public static void main(String[] args) throws Exception {
		Instances cancerInstances = loadData("cancer.txt");
		shuffleInstances(cancerInstances);
		shuffleInstances(cancerInstances);
		
		// Divide the data to training and test set - 80% training and 20% test
		Instances trainingSet = new Instances(cancerInstances);
		Instances testSet = new Instances(cancerInstances);
		trainingSet.clear();
		testSet.clear();
		int counter = 0;
		for (Instance instance : cancerInstances) {
			if(counter % 5 == 0){
				testSet.add(instance);
			} else {
				trainingSet.add(instance);
			}
			counter++;
		}
		
		Map<EvaluationTypes, Double> bestError = null;
		Kernel bestKernel = null;
		
		// Iterate optional polynomial kernels
		for (int polynomialKernelValue : new int[] { 2, 3, 4 }) {
			PolyKernel polyKernel = new PolyKernel();
			polyKernel.setExponent(polynomialKernelValue);
			Map<EvaluationTypes, Double> evaluation = trainAndAnalyzeKernel(polyKernel, trainingSet, testSet);
			
			System.out.println(MessageFormat.format("For PolyKernel with degree {0} the rates are:\nTPR = {1}\nFPR = {2}",
					polynomialKernelValue, evaluation.get(EvaluationTypes.TPR), evaluation.get(EvaluationTypes.FPR)));
			if(bestError == null || bestError.get(EvaluationTypes.BOTH) < evaluation.get(EvaluationTypes.BOTH)){
				bestError = evaluation;
				bestKernel = polyKernel;
			}
		}
		
		// Iterate optional RBF kernel values
		for (double rbfKernelValue : new double[] { 0.01, 0.1, 1 }) {
			RBFKernel rbfKernel = new RBFKernel();
			rbfKernel.setGamma(rbfKernelValue);
			Map<EvaluationTypes, Double> evaluation = trainAndAnalyzeKernel(rbfKernel, trainingSet, testSet);
			
			System.out.println(MessageFormat.format("For RBFKernel with gamma {0} the rates are:\nTPR = {1}\nFPR = {2}",
					rbfKernelValue, evaluation.get(EvaluationTypes.TPR), evaluation.get(EvaluationTypes.FPR)));
			if(bestError == null || bestError.get(EvaluationTypes.BOTH) < evaluation.get(EvaluationTypes.BOTH)){
				bestError = evaluation;
				bestKernel = rbfKernel;
			}
		}
		
		// Print best kernel
		String bestKernelStr = (bestKernel instanceof PolyKernel) ?
				"Poly with degree " + ((PolyKernel) bestKernel).getExponent() :
				"RBF with gamma " + ((RBFKernel) bestKernel).getGamma();
		System.out.println(MessageFormat.format("The best kernel is: {0} TPR={1}, FPR={2}",
				bestKernelStr, bestError.get(EvaluationTypes.TPR), bestError.get(EvaluationTypes.FPR)));
		
		/******************************************
		 * Second phase - finding the best C value: 
		 ******************************************/
		
		SVM svm = new SVM();
		svm.setKernel(bestKernel);
		Double bestC = null;
		bestError = null; 
		for(int i = 1; i >= -4; i--){
			for(int j = 3; j >= 1; j--){
				double c = Math.pow(10, i) * j / 3;
				svm.setC(c);
				svm.buildClassifier(trainingSet);
				int[] confusion = svm.calcConfusion(testSet);
				Map<EvaluationTypes, Double> errorData = extractErrorData(confusion);
				if(bestError == null || bestError.get(EvaluationTypes.BOTH) < errorData.get(EvaluationTypes.BOTH)){
					bestError = errorData;
					bestC = c;
				}
				System.out.println(MessageFormat.format("For C = {0} the rates are:\nTPR = {1}\nFPR = {2}",
						Double.toString(c), errorData.get(EvaluationTypes.TPR), errorData.get(EvaluationTypes.FPR)));
			}
		}
		System.out.print(MessageFormat.format("The best c is: {0} TPR={1}, FPR={2}",
				bestC, bestError.get(EvaluationTypes.TPR), bestError.get(EvaluationTypes.FPR)));
		
	}
	
	private static Map<EvaluationTypes, Double> trainAndAnalyzeKernel(Kernel kernel, Instances train, Instances test){
		SVM svm = new SVM();
		int[] confusion; // [TP, FP, TN, FN]
		try {
			svm.setKernel(kernel);
			svm.buildClassifier(train);
			confusion = svm.calcConfusion(test);
		} catch (Exception e) {
			System.err.println("Error setting SVM kernel. " + e);
			return null;
		}
		return extractErrorData(confusion);
	}
	
	private static Map<EvaluationTypes, Double> extractErrorData(int[] confusion){
		Map<EvaluationTypes, Double> results = new HashMap<>();
		results.put(EvaluationTypes.TPR, (double) confusion[0] / (confusion[0] + confusion[3]));// TP / (TP + FN)
		results.put(EvaluationTypes.FPR, (double) confusion[1] / (confusion[1] + confusion[2]));// FP / (FP + TN)
		results.put(EvaluationTypes.BOTH, results.get(EvaluationTypes.TPR) - ALPHA * results.get(EvaluationTypes.FPR));
		return results;
	}
	
	private static void shuffleInstances(Instances instances){
		instances.randomize(new Random());
	}
	
	public static enum EvaluationTypes { TPR, FPR, BOTH; } // BOTH = TPR - ALPGA * FPR
}
