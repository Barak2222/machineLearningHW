package HomeWork4;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

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
		Knn knn = new Knn();
		HyperParameters hp = knn.findBestHyperParameters(loadData("glass.txt"));
		System.out.println("[GLASS] Best Hyper Parameters: " + hp);
		
		knn = new Knn();
		hp = knn.findBestHyperParameters(loadData("cancer.txt"));
		System.out.println("[CANCER] Best Hyper Parameters: " + hp);
		
        //TODO: complete the Main method
	}

}
