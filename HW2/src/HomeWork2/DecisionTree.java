package HomeWork2;

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
   	boolean isLeaf;
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
		
		buildTree(instances);
		System.out.println("tree built:" + rules.size());

		if(m_pruningMode == PruningMode.Rule){
			rulePrunning(instances);
		}
	}

	private void buildTree(Instances instances) {
		
		// Initialize queue
		Queue<Node> queue = new LinkedBlockingQueue<>();
		
		// Create root
		rootNode = new Node();
		rootNode.isLeaf = false;
		Rule rootRule = new Rule();
		rootRule.basicRule = new ArrayList<>();
		rootNode.nodeRule = rootRule;
		rootNode.instances = instances;
		queue.add(rootNode);
		
		// Loop until queue is empty
		while(queue.size() > 0){
			Node nodeToProcess = queue.poll();
			if(nodeToProcess.isLeaf){
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
				if(currentInfoGain > 0 && currentInfoGain > bestInfoGain){
					bestInfoGain = currentInfoGain;
					bestAttributeIndex = i;
				}
			}
			
			// Case that 2 or more instances with same attribute values and different class
			if(bestAttributeIndex == -1){
				nodeToProcess.isLeaf = true;
				int[] countMajorityOfClasses = new int[2];
				for (Instance instance : nodeToProcess.instances) {
					countMajorityOfClasses[(int) instance.value(classIndex)]++;
				}
				nodeToProcess.nodeRule.returnValue = countMajorityOfClasses[0] > countMajorityOfClasses[1] ? 0.0 : 1.0;
				rules.add(nodeToProcess.nodeRule);
				continue;
			}
			// If chiSquare score is lower than 15.51, prune the branch and continue
			// to the next element in the queue
			if (m_pruningMode == PruningMode.Chi){
				double chiSquare = calcChiSquare(nodeToProcess.instances, bestAttributeIndex);
				if (chiSquare < 15.51)
					continue;
			}

			// Create children
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
			childNode.isLeaf = perfectlyClassifiedValue != null;
			if(childNode.isLeaf){
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
		
		// Iterate children and sum into sumOfChildEntropies
		for(int attributevalue = 0; attributevalue < NUMBER_OF_VALUES_FOR_ATTRIBUTE; attributevalue++){
			Instances instancesWithValue = filterInstancesWithAttributeValue(subsetOfTrainingData, attributeIndex, attributevalue);
			double entropyForCurrent = calcEntropyForInstances(instancesWithValue);
			sumOfChildEntropies += entropyForCurrent * instancesWithValue.size() / S_SIZE;
		}
		
		// Calculate informationGain
		double informationGain = entropyForS - sumOfChildEntropies;
		
		// Calculate "splitInformation"
		return informationGain;
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
				sum+= d * Math.log10(d);
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
	public double calcAvgError(Instances instances){
		int errorCount = 0;
		for (Instance instance : instances){
			if (classifyInstance(instance) != instance.value(classIndex))
				errorCount++;
		}
		return (double)errorCount / instances.size();
	}

	private void rulePrunning(Instances instances){
		boolean continuePrunning = true;
		while (continuePrunning){
			double errorBeforeRemoval = calcAvgError(validationSet);
			Rule ruleToRemove = null;
			double bestImprovement = -1;
			List<Rule> clonedRules = new ArrayList<>(rules);
			for(Rule rule: clonedRules){
				rules.remove(rule);
				double curError = calcAvgError(validationSet);
				double errorImprovement = errorBeforeRemoval - curError;
				if (errorImprovement > 0 && errorImprovement > bestImprovement){
					bestImprovement = errorImprovement;
					ruleToRemove = rule;
				}
				rules.add(rule);
			}
			if (ruleToRemove != null)
				rules.remove(ruleToRemove);
			else
				continuePrunning = false;
		}
	}

	private double calcChiSquare(Instances instancesSubset, int attributeIndex){
		int subsetSize = instancesSubset.size();
		int numOfVals = instancesSubset.attribute(attributeIndex).numValues();
		double chiSquare = 0;
		int pf, nf, Df;
		int y0 = 0, y1 = 0;
		double E0, E1;

		// Count instances by class value
		for (Instance instance : instancesSubset)
			// Y = 0
			if (instance.value(classIndex) == 0.0)
				y0++;
			// Y = 1
			else
				y1++;

		for(int attrValue = 0; attrValue < numOfVals; attrValue++){
			pf = 0; nf = 0; Df = 0;
			for (Instance instance1 : instancesSubset){
				if (instance1.value(attributeIndex) == (double)attrValue){
					// Df - number of instances for current attribute Value
					Df++;
					if (instance1.value(classIndex) == 0.0)
						// xj = f && Y = 0
						pf++;
					else
						// xj = f && Y = 1
						nf++;
				}
			}
			E0 = Df * ((double)y0 / subsetSize);
			E1 = Df * ((double)y1 / subsetSize);

			if (E0 != 0)
				chiSquare += Math.pow((pf - E0), 2) / E0;
			if (E1 != 0)
				chiSquare += Math.pow((nf - E1), 2) / E1;
			chiSquare += 0;
		}
		return chiSquare;
	}

	public void setPruningMode(PruningMode pruningMode) {
		m_pruningMode = pruningMode;
	}
	
	public void setValidation(Instances validation) {
		validationSet = validation;
	}
    
    @Override
	public double classifyInstance(Instance instance) {
    	
    	// Try to find rule that perfectly matches
    	for (Rule rule : rules) {
			if(isRuleApplied(rule, instance)){
				return rule.returnValue;
			}
		}
    	
    	// Find rule with largest amout of consequitive conditions
    	int longestRuleChainSize = -1;
    	List<Rule> rulesWithLongestChain = new ArrayList<>();
    	
    	for (Rule rule : rules) {
			int chainSize = calculateLongestRulesSequence(rule, instance);
			if(chainSize > longestRuleChainSize){
				rulesWithLongestChain = new ArrayList<>();
				longestRuleChainSize = chainSize;
			}
			if(chainSize == longestRuleChainSize){
				rulesWithLongestChain.add(rule);
			}
		}
    	if(rulesWithLongestChain.size() == 1){
    		return rulesWithLongestChain.get(0).returnValue;
    	}
    	
    	// Classify with the majority of return values of rules in rulesWithLongestChain
    	int firstClassCount = 0;
    	int secondClassCount = 0;
    	for (Rule rule : rulesWithLongestChain) {
			if(rule.returnValue == 0.0){
				firstClassCount++;
			} else {
				secondClassCount++;
			}
		}
    	return (firstClassCount > secondClassCount) ? 0.0 : 1.0;
	}
    
    private int calculateLongestRulesSequence(Rule rule, Instance instance) {
    	List<BasicRule> logicList = rule.basicRule;
		int count = 0;
    	for (BasicRule basicRule : logicList) {
			if(instance.value(basicRule.attributeIndex) != basicRule.attributeValue){
				return count;
			}
			count++;
		}
		return count;
	}
    
    public int getRulesCount(){
    	return rules.size();
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
