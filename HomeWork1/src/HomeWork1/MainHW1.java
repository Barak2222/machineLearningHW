package HomeWork1;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import weka.core.Instances;

public class MainHW1 {
	private static final String TRAINING_SET_PATH = "C:/cod/machineLearning/data/wind_training.txt";
	private static final String TESTING_SET_PATH = "C:/cod/machineLearning/data/wind_testing.txt";
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}
		
	/**
	 * Sets the class index as the last attribute.
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		return data;
	}
	
	public static void main(String[] args) throws Exception {
		//load data
		Instances trainingData = loadData(TRAINING_SET_PATH);
		Instances testingData = loadData(TESTING_SET_PATH);

		// Set class index
		trainingData.setClassIndex(trainingData.numAttributes() - 1);

		// Shuffle data
		Random random = new Random();
		trainingData.randomize(random);

		//train classifier
		LinearRegression linearRegressionClassifier = new LinearRegression();
		linearRegressionClassifier.buildClassifier(trainingData);

		testingData.setClassIndex(testingData.numAttributes() - 1);
		testingData.randomize(random);
	}

}
