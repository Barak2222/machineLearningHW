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
		
		// GLASS dta
		double error = knn.findBestHyperParameters(glassInstances);
		HyperParameters hp = knn.hyperParameters;
		System.out.println(MessageFormat.format(
				"Cross validation error with K = {0}, p = {1}, majority function = {2} for glass data is: {3}", 
				hp.k, hp.lpDistance, hp.majority, error));
		
		
		// CANCER data
		knn = new Knn();
		error = knn.findBestHyperParameters(cancerInstances);
		hp = knn.hyperParameters;
		System.out.println(MessageFormat.format(
				"Cross validation error with K = {0}, p = {1}, majority function = {2} for cancer data is: {3}", 
				hp.k, hp.lpDistance, hp.majority, error));
		double[] precisioAndRecall = knn.calcConfusion(cancerInstances, 10);
		System.out.println(MessageFormat.format("The average Precision for the cancer dataset is: {0}", precisioAndRecall[0]));
		System.out.println(MessageFormat.format("The average Recall for the cancer dataset is: {0}", precisioAndRecall[1]));
		
        //TODO: complete the Main method
	}

}
