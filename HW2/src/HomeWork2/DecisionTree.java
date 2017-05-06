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
			if(nodeToProcess.isPerfectlyClassified){
				continue;
			}
			Instances instancesThatGoToNode = instances;// TODO change this
			
			// Choose best attribute
			int bestAttributeIndex = -1;
			double bestInfoGain = -1;
			for(int i = 0; i < instances.numAttributes() - 1; i++){
				double currentInfoGain = calcInfoGain(instancesThatGoToNode, i);
				if(currentInfoGain > bestAttributeIndex){
					bestInfoGain = currentInfoGain;
					bestAttributeIndex = i;
				}
			}
			
			nodeToProcess.children = buildChildren(nodeToProcess, bestAttributeIndex, instancesThatGoToNode);
			// create children
			
			
		}
		// TODO Auto-generated method stub
		
	}
	
	private Node[] buildChildren(Node parent, int attributeIndex, Instances instances) {
		parent.attributeIndex = attributeIndex;
		List<BasicRule> listOfParentsRules = parent.nodeRule.basicRule;
		
		// eg number of children
		final int NUMBER_OF_VALUES_FOR_ATTRIBUTE = instances.attribute(attributeIndex).numValues();
		Node[] children = new Node[NUMBER_OF_VALUES_FOR_ATTRIBUTE];
		for(int i = 0; i < NUMBER_OF_VALUES_FOR_ATTRIBUTE; i++){
			Node node = new Node();
			Rule rule = new Rule();
			node.parent = parent;
			
			List<BasicRule> listOfChildRules = new ArrayList<>(listOfParentsRules);
			listOfChildRules.add(new BasicRule(attributeIndex, i));
			rule.basicRule = listOfChildRules;
			node.nodeRule = rule;
			
			Double perfectlyClassifiedValue = isPerfectlyClassified(instances, rule);
			if(perfectlyClassifiedValue != null){
				rule.returnValue = perfectlyClassifiedValue;
				node.isPerfectlyClassified = true;
			}
			children[i] = node;
		}
		
		return children;
	}

	private Double isPerfectlyClassified(Instances instances, Rule r) {
		// TODO Auto-generated method stub
		return null;
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
			childrenProbabilities[attributevalue] = instancesWithValue.size() / S_SIZE;
		}
		
		// Calculate informationGain
		double informationGain = entropyForS - sumOfChildEntropies;
		
		// Calculate "splitInformation"
		double splitInformation = calcEntropy(childrenProbabilities);
		return informationGain / splitInformation;
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
	
//	private static Instances fildterInstancesWithAttributeValue(Instances instances, int attributeIndex, double attributevalue){
//		try {
//			RemoveWithValues filter = new RemoveWithValues();
//			String[] options = new String[4];
//			options[0] = "-C";   // Choose attribute to be used for selection
//			options[1] = Integer.toString(attributeIndex); // Attribute number    
//			options[2] = "-L";
//			options[3] = Integer.toString((int) attributevalue);
//			filter.setOptions(options);
//			filter.setInputFormat(instances);
//			return Filter.useFilter(instances, filter);
//		} catch (Exception e) {
//			e.printStackTrace();
//		}
//		throw new IllegalArgumentException();
//	}
//	
	private static Instances filterInstancesWithAttributeValue(Instances source, int attributeIndex, double attributevalue){
		Instances clone = new Instances(source);
		for (int i = source.size() - 1; i >= 0; i--) {
			Instance instance = source.get(i);
			double attrValueForInstance = instance.value(attributeIndex);
			if( attrValueForInstance != attributevalue){
				clone.delete(i);
			}
		}
		return clone;
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
