package HomeWork5;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.text.MessageFormat;
import java.util.Random;

import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Instances;

public class MainHW5 {

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
		SVM svm;
		int polynomialKernelValues[] = { 2, 3, 4 };
		double RBFKernelValues[] = { 0.01, 0.1, 1 };
		int TPR = 1;
		int FPR = 1; 
		
		// TODO:
		// Divide the data to training and test set - 80% training and 20% test
		Instances trainingSet = null;
		Instances testSet = null;
		
		// For each kernel value, build the SVM classifier on the training set using the SMO WEKA class
		for (int i = 0; i < polynomialKernelValues.length; i++) {
			for (int j = 0; j < RBFKernelValues.length; j++) {
				svm = new SVM();
				
				// Set kernel to be polynomial
				PolyKernel polyKernel = new PolyKernel();
				polyKernel.setExponent(i);
				svm.setKernel(polyKernel);
				svm.buildClassifier(trainingSet);
				
				// TODO: Calculate TPR and FPR for poly
				TPR = 0;
				FPR = 0;
				
				System.out.println(
						MessageFormat.format("For PolyKernel with degree {0} the rates are:\nTPR = {1}\nFPR = {2}", i, TPR, FPR));
				
				// Set kernel to be RBF
				RBFKernel rbfKernel = new RBFKernel();
				rbfKernel.setGamma(j);
				svm.setKernel(rbfKernel);
				svm.buildClassifier(trainingSet);
				
				// TODO: Calculate TPR and FPR for poly
				TPR = 0;
				FPR = 0;
				
				System.out.println(
						MessageFormat.format("For RBFKernel with gamma {0} the rates are:\nTPR = {1}\nFPR = {2}", j, TPR, FPR));

			}
		}
		// TODO: Check what's the best kernel (poly or RBF), and save the best TPR & FPR
		System.out.println(
				MessageFormat.format("The best kernel is: <Poly or RBF> <kernel parameter (degree or gamma)> <TPR - FPR>"));
		
		
	}
}
