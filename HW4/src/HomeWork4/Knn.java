package HomeWork4;

import java.text.MessageFormat;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;

import HomeWork4.Knn.HyperParameters.Majority;
import HomeWork4.Knn.HyperParameters.LPDistanceOptions;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class Knn implements Classifier {
	
	public enum EditMode {None, Forwards, Backwards};
	private EditMode m_editMode = EditMode.None;
	private Instances m_trainingInstances;
	private int numberOfClasses;
	public HyperParameters hyperParameters;

	public EditMode getEditMode() {
		return m_editMode;
	}

	public void setEditMode(EditMode editMode) {
		m_editMode = editMode;
	}
	
	public HyperParameters findBestHyperParameters(Instances instances){
		shuffleInstances(instances);
		numberOfClasses = instances.attribute(instances.classIndex()).numValues();
		List<HyperParameters> options  = HyperParameters.getHyperParametersPermutations();
		double bestError = Double.MAX_VALUE;
		HyperParameters bestHyperParameters = null;
		for (HyperParameters hyperParameters : options) {
			this.hyperParameters = hyperParameters;
			double error = crossValidationError(instances, 10);
			if(bestError == Double.MAX_VALUE || error < bestError){
				bestError = error;
				bestHyperParameters = hyperParameters;
			}
		}
		this.hyperParameters = bestHyperParameters;
		return bestHyperParameters;
	}

	/**
	 * Builds a kNN from the training data.
	 */
	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		shuffleInstances(arg0);
		numberOfClasses = arg0.attribute(arg0.classIndex()).numValues();
		switch (m_editMode) {
		case None:
			noEdit(arg0);
			break;
		case Forwards:
			editedForward(arg0);
			break;
		case Backwards:
			editedBackward(arg0);
			break;
		default:
			noEdit(arg0);
			break;
		}
	}
	
	/***
	 * Returns the classification of the instance
	 * Input: Instance Object
	 * Output: double number, represent the classified class
	 */
	@Override
	public double classifyInstance(Instance instance) {
		Map<Instance, Double> neighbors = findNearestNeighbors(instance);
		if(hyperParameters.majority == Majority.weighted){
			return getWeightedClassVoteResult(neighbors);
		}
		return getClassVoteResult(neighbors);
	}
	
	/**
	 * Store the training set in the m_trainingInstances using the forwards editing
	 * @param instances
	 */
	private void editedForward(Instances instances) {
		//TODO: implement this method
	}

	/**
	 * Store the training set in the m_trainingInstances using the backwards editing.
	 * @param instances
	 */
	private void editedBackward(Instances instances) {
		//TODO: implement this method
	}
	
	/**
	 * Calculate the average error on a given instances set. The average error is the total number of classification mistakes on the input 
	 * instances set and divides that by the number of instances in the input set.
	 * @param instances instances
	 * @return classification error
	 */
	private double calcAvgError(Instances instances){
		int numOfErrors = 0;
		for (Instance instance : instances) {
			double predictedClass = classifyInstance(instance);
			double actualClass = instance.value(instance.classIndex());
			if(predictedClass != actualClass){
				numOfErrors++;
			}
		}
		return ((double) numOfErrors) / instances.size(); 
	}
	
	/**
	 * Calculate the Precision & Recall on a given instances set.
	 * @param instances instances
	 * @return double array of size 2. First index for Precision and the second for Recall.
	 */
	private double[] calcConfusion(Instances instances) {
		// TODO;
		return null;
	}

	/**
	 * Calculate the cross validation error = average error on all folds.
	 * @param instances instances
	 * @param numOfFolds number of folds
	 * @return Average fold error (double)
	 */
	private double crossValidationError(Instances instances, int numOfFolds) {
		double sumOfErrors = 0;
		for(int foldNumber = 0; foldNumber < numOfFolds; foldNumber++){
			
			// Split to training and validation
			Instances training = new Instances(instances);
			Instances validation = new Instances(instances);
			training.clear();
			validation.clear();
			for(int i = 0; i < instances.size(); i++){
				if(i / numOfFolds == foldNumber){
					validation.add(instances.get(i));
				} else {
					training.add(instances.get(i));
				}
			}
			
			// Calc error
			this.m_trainingInstances = training;
			double error = calcAvgError(validation);
			sumOfErrors+= error;
			
		}
		
		return sumOfErrors / numOfFolds;
	}
	
	/**
	 * Find the K nearest neighbors for the instance being classified(target).
	 * @param target input instance
	 * @return k nearest neighbors and their distance from the given instance
	 */
	private Map<Instance, Double> findNearestNeighbors(Instance target){
		Map<Instance, Double> result = new HashMap<>();
		Comparator<Instance> distanceComparator = new Comparator<Instance>() {
			@Override
			public int compare(Instance o1, Instance o2) {
				double d1 = distance(target, o1);
				double d2 = distance(target, o2);
				return Double.compare(d1, d2) * (-1);
			}
		};
		
		// Initialize priority queue of size K with the relevant comparision function
		PriorityQueue<Instance> queue = new PriorityQueue<>(hyperParameters.k, distanceComparator);
		for (Instance neighbor : m_trainingInstances) {
			queue.add(neighbor);
		}
		
		// Process data
		for (Instance closestNeighbor : queue) {
			result.put(closestNeighbor, distance(closestNeighbor, target));
		}
		return result;
	}
	
	/**
	 * Calculate the majority class of the neighbors
	 * @param nearestNeighbors The k nearest neighbors
	 * @return the majority vote on the class of the neighbors
	 */
	private double getClassVoteResult(Map<Instance, Double> nearestNeighborsWithDistances){
		if(nearestNeighborsWithDistances.isEmpty()){
			throw new IllegalArgumentException();
		}
		int[] countersArr = new int[numberOfClasses];
		for (Instance neighbor : nearestNeighborsWithDistances.keySet()) {
			countersArr[(int) neighbor.value(neighbor.classIndex())]++;
		}
		
		double bestClass = -1.0;
		int highestVotes = -1;
		for (int i = 0; i < countersArr.length; i++) {
			if(countersArr[i] > highestVotes){
				highestVotes = countersArr[i];
				bestClass = (double) i;
			}
		}
		return bestClass;
	}
	
	/**
	 * Calculate the weighted majority class of the neighbors. In this method the class vote is normalized by the distance
	 * from the instance being classified Instead of giving one vote to every class,
	 * you give a vote of 1/(distance from instance)^2
	 * @param nearestNeighbors A set of K nearest neighbors and their distances
	 * @return the majority vote on the class of the neighbors, where each neighbor's class is weighted by the neighbor’s
	 * distance from the instance being classified.
	 */
	private double getWeightedClassVoteResult(Map<Instance, Double> nearestNeighborsWithDistances){
		if(nearestNeighborsWithDistances.isEmpty()){
			throw new IllegalArgumentException();
		}
		double[] weightsArr = new double[numberOfClasses];
		for (Instance neighbor : nearestNeighborsWithDistances.keySet()) {
			int classIndex = (int) neighbor.value(neighbor.classIndex());
			weightsArr[classIndex]+= (double) 1/(Math.pow(nearestNeighborsWithDistances.get(neighbor), 2));
		}
		
		double bestClass = -1.0;
		double highestweight = -1;
		for (int i = 0; i < weightsArr.length; i++) {
			if(weightsArr[i] > highestweight){
				highestweight = weightsArr[i];
				bestClass = (double) i;
			}
		}
		return bestClass;
	}
	
	/**
	 * Calculates the input instances’ distance according to the current distance function.
	 * @param a instance1
	 * @param b instance2
	 * @return the input instances’ distance according to the distance function that your algorithm is configured to use.
	 */
	private double distance(Instance a, Instance b){
		switch(hyperParameters.lpDistance){
			case one:      { return OnePDistance(a, b, 1); }
			case two:      { return OnePDistance(a, b, 2); }
			case three:    { return OnePDistance(a, b, 3); }
			case infinity: { return OneInfinityDistance(a, b);}
			default:       { throw new IllegalArgumentException();}
		}
	}
	
	/**
	 * 
	 * @param a instance1
	 * @param b instaqnce2
	 * @param pDistance p parameter to use
	 * @return the l-p distance between the two instances
	 */
	private double OnePDistance(Instance a, Instance b, int pDistance){
		double sum = 0;
		for(int attr = 0; attr < a.numAttributes() - 1; attr++){
//			sum+= TODO
		}
		
		
		return Math.pow(sum, (1/pDistance));
	}
	
	/**
	 * 
	 * @param a instance1
	 * @param b instance2
	 * @return the l-infinity distance between two instances
	 */
	private double OneInfinityDistance(Instance a, Instance b){
		// TODO
		return 0;
	}
	
	private void shuffleInstances(Instances instances){
		instances.randomize(new Random());
	}
	
	/**
	 * Store the training set in the m_trainingInstances without editing.
	 * @param instances instances
	 */
	private void noEdit(Instances instances) {
		m_trainingInstances = new Instances(instances);
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		return null;
	}
	
	public static class HyperParameters {
		public enum Majority { uniform, weighted;}
		public enum LPDistanceOptions { one, two, three, infinity }
		
		public int k;
		public LPDistanceOptions lpDistance;
		public Majority majority;
		
		public static List<HyperParameters> getHyperParametersPermutations (){
			List<HyperParameters> result = new ArrayList<>();
			for(Majority majority: Majority.values()){
				for(LPDistanceOptions pDistance: LPDistanceOptions.values()){
					for(int k = 1; k <= 20; k++){
						HyperParameters hp = new HyperParameters();
						hp.k = k;
						hp.lpDistance = pDistance;
						hp.majority = majority;
						result.add(hp);
					}
				}
			}
			return result;
		}
		
		public String toString(){
			return MessageFormat.format("k: {0}, 1-p Distance: {1}, Majority: {2}", this.k, this.lpDistance, this.k);
		}
	}
	
//	static class InstancesPriorityQueue{
//		private PriorityQueue queue;
//		private Instance target;
//		public InstancesPriorityQueue(Instance target, int size){
//			this.target = target;
//			queue = new PriorityQueue<>(size, new Comparator<Instance>() {
//			})
//		}
//	}
}
