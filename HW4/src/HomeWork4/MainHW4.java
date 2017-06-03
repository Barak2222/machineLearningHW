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
		HyperParameters hp = knn.findBestHyperParameters(glassInstances);
		System.out.println("[GLASS] Best Hyper Parameters: " + hp);
		
		knn = new Knn();
		hp = knn.findBestHyperParameters(cancerInstances);
		System.out.println("[CANCER] Best Hyper Parameters: " + hp);
		double[] precisioAndRecall = knn.calcConfusion(cancerInstances, 10);
		System.out.println(MessageFormat.format("[CANCER] precision {0} recall {1}", precisioAndRecall[0], precisioAndRecall[1]));
		
        //TODO: complete the Main method
	}

}
