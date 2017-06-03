package HomeWork4;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.text.MessageFormat;

import HomeWork4.Knn.EditMode;
import HomeWork4.Knn.HyperParameters;
import weka.core.Instances;

public class MainHW4 {

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
		Instances glassInstances = loadData("glass.txt");
		Instances cancerInstances = loadData("cancer.txt");
		Knn knn = new Knn();
		knn.setEditMode(EditMode.None);
			
        // ---- PHASE 1 ----
		
		// GLASS dta
		double error = knn.findBestHyperParameters(glassInstances);
		HyperParameters hpForGlass = knn.hyperParameters;
		System.out.println(MessageFormat.format(
				"Cross validation error with K = {0}, p = {1}, majority function = {2} for glass data is: {3}", 
				hpForGlass.k, hpForGlass.lpDistance, hpForGlass.majority, error));
		
		
		// CANCER data
		knn = new Knn();
		error = knn.findBestHyperParameters(cancerInstances);
		HyperParameters hpForCancer = knn.hyperParameters;
		System.out.println(MessageFormat.format(
				"Cross validation error with K = {0}, p = {1}, majority function = {2} for cancer data is: {3}", 
				hpForCancer.k, hpForCancer.lpDistance, hpForCancer.majority, error));
		double[] precisioAndRecall = knn.calcConfusion(cancerInstances, 10);
		System.out.println(MessageFormat.format("The average Precision for the cancer dataset is: {0}", precisioAndRecall[0]));
		System.out.println(MessageFormat.format("The average Recall for the cancer dataset is: {0}", precisioAndRecall[1]));
		
        // ---- PHASE 2 ----
		
		knn.hyperParameters = hpForGlass;
		int[] foldParametersToCheck = {3, 5, 10, 50, glassInstances.size()};
		for (int numberOfFolds : foldParametersToCheck) {
			System.out.println("----------------------------");
			System.out.println(MessageFormat.format("Results for {0} folds:", numberOfFolds));
			System.out.println("----------------------------");
			
			for (EditMode editMode : EditMode.values()) {
				
				// Run cross validation
				knn.setEditMode(editMode);
				error = knn.crossValidationError(glassInstances, numberOfFolds);
				
				// Print
				long totalElapsedTimeInNanoseconds = knn.timeToClassifyInstancesInAllFolds;
				long averageElapseTimeForEachFold = totalElapsedTimeInNanoseconds / numberOfFolds;
				System.out.println(MessageFormat.format(
						"Cross validation error of {0}-Edited knn on glass dataset is {1} and the average elapsed time is {2}",
						editMode, error, averageElapseTimeForEachFold));
				System.out.println(MessageFormat.format("The total elapsed time is: {0}", totalElapsedTimeInNanoseconds));
				System.out.println(MessageFormat.format("The total number of instances used in the classification phase is: {0}", knn.trainingInstancesCount));				
			}
		}
	}
}
