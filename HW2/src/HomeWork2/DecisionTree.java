package HomeWork2;

import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.LinkedBlockingQueue;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;

class BasicRule {
    int attributeIndex;
	int attributeValue;
}

class Rule {
   	List<BasicRule> basicRule;
   	double returnValue;
}

class Node {
	Node[] children;
	Node parent;
	int attributeIndex;
	double returnValue;
   	Rule nodeRule = new Rule();
   	boolean isPerfectlyClassified;
}

public class DecisionTree implements Classifier {
	private Node rootNode;
	public enum PruningMode {None, Chi, Rule};
	private PruningMode m_pruningMode;
   	Instances validationSet;
   	private List<Rule> rules = new ArrayList<Rule>();
   	private int classIndex;
   	
	@Override
	public void buildClassifier(Instances instances) throws Exception {
		classIndex = 10;
		
		// TODO
		
		// preProcessing
		
		buildTree(instances);
		
		// postProcessing
	}

	private void buildTree(Instances instances) {
		Queue<Node> queue = new LinkedBlockingQueue<>();
		rootNode = new Node();
		rootNode.isPerfectlyClassified = false;
		queue.add(rootNode);
		while(queue.size() > 0){
			Node nodeToProcess = queue.poll();
			
//			ArrayList<Instance> instancesForNode = new ArrayList<Instance>();
//			for(int i = 0; i < instances.size(); i++){
//				Instance currentInstance = instances.get(i);
//				if(isRuleApplied(nodeToProcess.nodeRule, currentInstance)){
//					instancesForNode.add(currentInstance);
//				}
//			}
			
			// if perfectly calssified
			if(! nodeToProcess.isPerfectlyClassified){
				continue;
			}
			
			// Choose best attribute
			
		}
		// TODO Auto-generated method stub
		
	}
	
	private double calcInfoGain(Instances subsetOfTrainingData, int attributeIndex){
		final int S_SIZE = subsetOfTrainingData.size();
		double[] numberOfInstancesForClass = new double[2];
		
		// First calculate entropy for S (= the subset)
		for (Instance instance : subsetOfTrainingData) {
			numberOfInstancesForClass[(int) instance.value(classIndex)] ++;
		}
		
		for (int i = 0; i < 2; i++) {
			numberOfInstancesForClass[i] /= S_SIZE;
		}
		double entropyForS = calcEntropy(numberOfInstancesForClass);

		double SumOfSmallEntropies = 0;
		final int NUMBER_OF_VALUES_FOR_ATTRIBUTE = subsetOfTrainingData.attribute(attributeIndex).numValues();

		for(int attributevalue = 0; attributevalue < NUMBER_OF_VALUES_FOR_ATTRIBUTE; attributevalue++){
			Instances instancesToProcess = filterInstancesWithAttributeValue(subsetOfTrainingData, attributeIndex, attributevalue);
			double entropyForCurrent = 0;
			
			double[] numberOfInstancesForClass = new double[2];
			
			// First calculate entropy for S (= the subset)
			for (Instance instance : subsetOfTrainingData) {
				numberOfInstancesForClass[(int) instance.value(classIndex)] ++;
			}
		}
		
		int numOfInstancesInFirstCalss = 0;
		int numOfInstancesInSecondCalss = 0;
		
		// Iterate relevant instances
		for (Instance instance : subsetOfTrainingData) {
			instance.attribute(attributeIndex);
			
			
		}
		double[] probablilities = {numOfInstancesInFirstCalss / S_SIZE, numOfInstancesInSecondCalss / S_SIZE};
		
		return -1;
	}
	
	private double calcEntropyForInstances(Instances instances){
		double[] numberOfInstancesForClass = new double[2];
		for (Instance instance : instances) {
			numberOfInstancesForClass[(int) instance.value(classIndex)] ++;
		}
		for (int i = 0; i < 2; i++) {
			numberOfInstancesForClass[i] /= instances.size();
		}
		return calcEntropy(numberOfInstancesForClass);
	}
	
	private static Instances filterInstancesWithAttributeValue(Instances instances, int attributeIndex, double attributevalue){
		try {
			RemoveWithValues filter = new RemoveWithValues();
			String[] options = new String[4];
			options[0] = "-C";   // Choose attribute to be used for selection
			options[1] = Integer.toString(attributeIndex); // Attribute number    
			options[2] = "-L";
			options[3] = Integer.toString((int) attributevalue);
			filter.setOptions(options);
			filter.setInputFormat(instances);
			return Filter.useFilter(instances, filter);
		} catch (Exception e) {
			e.printStackTrace();
		}
		throw new IllegalArgumentException();
	}
	
	private double calcEntropy(double[] probabilities){
		double sum = 0;
		for (double d : probabilities) {
			sum+= d * Math.log(d);
		}
		return (-1) * sum;
	}
	
	private boolean isRuleApplied(Rule rule, Instance instance){
		
		return false;
	}

	public void setPruningMode(PruningMode pruningMode) {
		m_pruningMode = pruningMode;
	}
	
	public void setValidation(Instances validation) {
		validationSet = validation;
	}
    
    @Override
	public double classifyInstance(Instance instance) {
		//TODO: implement this method
    	return 0;
	}
    
    @Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}

}
