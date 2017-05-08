package HomeWork2;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import HomeWork2.DecisionTree.PruningMode;
import weka.core.Instances;

public class MainHW2 {

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
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	

	public static void main(String[] args) throws Exception {
		Instances trainingCancer = loadData("cancer_train.txt");
		Instances testingCancer = loadData("cancer_test.txt");
		Instances validationCancer = loadData("cancer_validation.txt");
		
		DecisionTree decisionTree = new DecisionTree();
		decisionTree.setPruningMode(PruningMode.None);
		decisionTree.buildClassifier(trainingCancer);
		
		int x = 1;
		
		System.out.println("Decision Tree with No pruning");
		System.out.println("The average train error of the decision tree is " + decisionTree.calcAvgError(trainingCancer));
		System.out.println("The average test error of the decision tree is " + decisionTree.calcAvgError(testingCancer));
		System.out.println("The amount of rules generated from the tree " + decisionTree.getRulesCount());
		
		decisionTree = new DecisionTree();
		decisionTree.setPruningMode(PruningMode.Chi);
		decisionTree.buildClassifier(trainingCancer);
		System.out.println("Decision Tree with Chi pruning");
		System.out.println("The average train error of the decision tree with Chi pruning is " + decisionTree.calcAvgError(trainingCancer));
		System.out.println("The average test error of the decision tree with Chi pruning is " + decisionTree.calcAvgError(testingCancer));
		System.out.println("The amount of rules generated from the tree " + decisionTree.getRulesCount());
		
		decisionTree = new DecisionTree();
		decisionTree.validationSet = validationCancer;
		decisionTree.setPruningMode(PruningMode.Rule);
		decisionTree.buildClassifier(trainingCancer);
		System.out.println("Decision Tree with Rule pruning");
		System.out.println("The average train error of the decision tree with Rule pruning is " + decisionTree.calcAvgError(trainingCancer));
		System.out.println("The average test error of the decision tree with Rule pruning is " + decisionTree.calcAvgError(testingCancer));
		System.out.println("The amount of rules generated from the tree " + decisionTree.getRulesCount());

		
	}
}
