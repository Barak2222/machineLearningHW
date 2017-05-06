package HomeWork2;

import java.rmi.UnexpectedException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.LinkedBlockingQueue;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

class BasicRule {
	public BasicRule(){}
	public BasicRule(int attrIdx, int attrVal){
		attributeIndex = attrIdx;
		attributeValue = attrVal;
	}
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
   	Instances instances;
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
		classIndex = 9;
		
		// preProcessing
		
		buildTree(instances);
		System.out.println("tree built");
		// postProcessing
	}

	private void buildTree(Instances instances) {
		
		// Initialize queue
		Queue<Node> queue = new LinkedBlockingQueue<>();
		
		// Create root
		rootNode = new Node();
		rootNode.isPerfectlyClassified = false;
		Rule rootRule = new Rule();
		rootRule.basicRule = new ArrayList<>();
		rootNode.nodeRule = rootRule;
		rootNode.instances = instances;
		queue.add(rootNode);
		
		// Loop until queue is empty
		while(queue.size() > 0){
			Node nodeToProcess = queue.poll();
			if(nodeToProcess.isPerfectlyClassified){
				nodeToProcess.instances = null; // Free memory
				
				// Node is leaf => add rule
				rules.add(nodeToProcess.nodeRule);
				continue;
			}
			
			// Choose best attribute
			int bestAttributeIndex = -1;
			double bestInfoGain = -1;
			for(int i = 0; i < instances.numAttributes() - 1; i++){
				double currentInfoGain = calcInfoGain(nodeToProcess.instances, i);
				if(currentInfoGain > bestInfoGain){
					bestInfoGain = currentInfoGain;
					bestAttributeIndex = i;
				}
			}
			
			// Case that node could not be further splitted (2 instances with same x and different class
			if(bestAttributeIndex == -1){
				nodeToProcess.instances = null; // Free memory
				continue;
			}
			
			nodeToProcess.children = buildChildren(nodeToProcess, bestAttributeIndex);
			if(nodeToProcess != rootNode){
				nodeToProcess.instances = null; // Free memory
			}
			queue.addAll(Arrays.asList(nodeToProcess.children));
		}
	}
	
	private Node[] buildChildren(Node parent, int attributeIndex) {
		parent.attributeIndex = attributeIndex;
		List<BasicRule> listOfParentsRules = parent.nodeRule.basicRule;
		
		// eg number of children
		final int NUMBER_OF_VALUES_FOR_ATTRIBUTE = rootNode.instances.attribute(attributeIndex).numValues();
		List<Node> children = new ArrayList<>();
		for(int attributeValue = 0; attributeValue < NUMBER_OF_VALUES_FOR_ATTRIBUTE; attributeValue++){
			
			// Create new child
			Node childNode = new Node();
			childNode.parent = parent;
			childNode.instances = filterInstancesWithAttributeValue(parent.instances, attributeIndex, attributeValue);
			if(childNode.instances.isEmpty()){
				continue;
			}
			
			// Create relevant rule
			Rule rule = new Rule();
			List<BasicRule> listOfChildRules = new ArrayList<>(listOfParentsRules);
			listOfChildRules.add(new BasicRule(attributeIndex, attributeValue));
			rule.basicRule = listOfChildRules;
			childNode.nodeRule = rule;
			
			// Check if node is a leaf
			Double perfectlyClassifiedValue = isPerfectlyClassified(childNode.instances);
			childNode.isPerfectlyClassified = perfectlyClassifiedValue != null;
			if(childNode.isPerfectlyClassified){
				rule.returnValue = perfectlyClassifiedValue;
			}
			children.add(childNode);
		}
		
		Node[] result = new Node[children.size()];
		return children.toArray(result);
	}

	// true   =>    returns class value
	// false  =>    returns null
	private Double isPerfectlyClassified(Instances instances) {
		if(instances.isEmpty()){
			return null;
		}
		double expectedValue = instances.firstInstance().value(classIndex);
		for (Instance instance : instances) {
			if(instance.value(classIndex) != expectedValue){
				return null;
			}
		}
		return expectedValue;
	}

	private double calcInfoGain(Instances subsetOfTrainingData, int attributeIndex){
		final int S_SIZE = subsetOfTrainingData.size();
		
		// First calculate entropy for S (= the subset)
		double entropyForS = calcEntropyForInstances(subsetOfTrainingData);
		double sumOfChildEntropies = 0;
		
		// eg number of children
		final int NUMBER_OF_VALUES_FOR_ATTRIBUTE = subsetOfTrainingData.attribute(attributeIndex).numValues();
		double[] childrenProbabilities = new double[NUMBER_OF_VALUES_FOR_ATTRIBUTE];
		
		// Iterate children and sum into sumOfChildEntropies
		for(int attributevalue = 0; attributevalue < NUMBER_OF_VALUES_FOR_ATTRIBUTE; attributevalue++){
			Instances instancesWithValue = filterInstancesWithAttributeValue(subsetOfTrainingData, attributeIndex, attributevalue);
			double entropyForCurrent = calcEntropyForInstances(instancesWithValue);
			sumOfChildEntropies += entropyForCurrent * instancesWithValue.size() / S_SIZE;
			
			// Aggregate data for splitInformation
			childrenProbabilities[attributevalue] = (double) instancesWithValue.size() / S_SIZE;
		}
		// Calculate informationGain
		double informationGain = entropyForS - sumOfChildEntropies;
		
		// Calculate "splitInformation"
		double splitInformation = calcEntropy(childrenProbabilities);
		return informationGain / splitInformation;
	}
	
	private double calcEntropyForInstances(Instances instances){
		if (instances.size() == 0)
			return 0.0;
		double[] numberOfInstancesForClass = new double[2];
		for (Instance instance : instances) {
			numberOfInstancesForClass[(int) instance.value(classIndex)] ++;
		}
		for (int i = 0; i < 2; i++) {
			numberOfInstancesForClass[i] /= instances.size();
		}
		return calcEntropy(numberOfInstancesForClass);
	}

	private static Instances filterInstancesThatApplyToRule(Instances source, Rule rule){
		Instances clone = new Instances(source);
		for (int i = source.numInstances() - 1; i >= 0; i--) {
			Instance instance = clone.get(i);
			if(! isRuleApplied(rule, instance)){
				clone.delete(i);
			}
		}
		return clone;
	}
	
	private static Instances filterInstancesWithAttributeValue(Instances source, int attributeIndex, double attributeValue){
		Instances clone = new Instances(source);
		for (int i = source.numInstances() - 1; i >= 0; i--) {
			Instance instance = clone.get(i);
			double attrValueForInstance = instance.value(attributeIndex);
			if( attrValueForInstance != attributeValue){
				clone.delete(i);
			}
		}
		return clone;
	}
	
	private double calcEntropy(double[] probabilities){
		double sum = 0;
		for (double d : probabilities) {
			if (d != 0)
				sum+= d * Math.log(d);
		}
		return (-1) * sum;
	}
	
	private static boolean isRuleApplied(Rule rule, Instance instance){
		List<BasicRule> logicList = rule.basicRule;
		for (BasicRule basicRule : logicList) {
			if(instance.value(basicRule.attributeIndex) != basicRule.attributeValue){
				return false;
			}
		}
		return true;
	}

	public void setPruningMode(PruningMode pruningMode) {
		m_pruningMode = pruningMode;
	}
	
	public void setValidation(Instances validation) {
		validationSet = validation;
	}
    
    @Override
	public double classifyInstance(Instance instance) {
    	for (Rule rule : rules) {
			if(isRuleApplied(rule, instance)){
				return rule.returnValue;
			}
		}
    	throw new IllegalArgumentException();
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
