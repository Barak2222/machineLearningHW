package HomeWork4;

import java.text.MessageFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import HomeWork4.Knn.HyperParameters.PDistanceOptions;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class Knn implements Classifier {
	
	public enum EditMode {None, Forwards, Backwards};
	private EditMode m_editMode = EditMode.None;
	private Instances m_trainingInstances;
	public HyperParameters hyperParameters;

	public EditMode getEditMode() {
		return m_editMode;
	}

	public void setEditMode(EditMode editMode) {
		m_editMode = editMode;
	}
	
	public HyperParameters findBestHyperParameters(Instances instances){
		shuffleInstances(instances);
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
		return 0;
		//TODO: implement this method
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
	 * : Calculate the average error on a given instances set. The average error is the total number of classification mistakes on the input 
	 * instances set and divides that by the number of instances in the input set.
	 * @param instances instances
	 * @return classification error
	 */
	private double calcAvgError(Instances instances){
		// TODO
		return 0;
		
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
		// TODO
		return 0;
	}
	
	/**
	 * the K nearest neighbors for the instance being classified.
	 * @param instance input instance
	 * @return K nearest neighbors and their distance from the given instance
	 */
	private Map<Instance, Double> findNearestNeighbors(Instance instance){
		// TODO
		return null;
	}
	
	/**
	 * Calculate the majority class of the neighbors
	 * @param nearestNeighbors The k nearest neighbors
	 * @return the majority vote on the class of the neighbors
	 */
	private double getClassVoteResult(Map<Instance, Double> nearestNeighbors){
		// TODO
		return 0;
	}
	
	/**
	 * Calculate the weighted majority class of the neighbors. In this method the class vote is normalized by the distance
	 * from the instance being classified Instead of giving one vote to every class,
	 * you give a vote of 1/(distance from instance)^2
	 * @param nearestNeighbors A set of K nearest neighbors and their distances
	 * @return the majority vote on the class of the neighbors, where each neighbor's class is weighted by the neighbor’s
	 * distance from the instance being classified.
	 */
	private double getWeightedClassVoteResult(Map<Instance, Double> nearestNeighbors){
		// TODO
		return 0;
	}
	
	/**
	 * Calculates the input instances’ distance according to the current distance function.
	 * @param a instance1
	 * @param b instance2
	 * @return the input instances’ distance according to the distance function that your algorithm is configured to use.
	 */
	private double distance(Instance a, Instance b){
		// TODO => should use the fillowing two methods: OnePDistance, OneInfinityDistance
		return 0;
	}
	
	/**
	 * 
	 * @param a instance1
	 * @param b instaqnce2
	 * @param pDistance p parameter to use
	 * @return the l-p distance between the two instances
	 */
	private double OnePDistance(Instance a, Instance b, PDistanceOptions pDistance){
		// TODO
		return 0;
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
		public enum PDistanceOptions { one, two, three, infinity }
		
		public int k;
		public PDistanceOptions pDistance;
		public Majority majority;
		
		public static List<HyperParameters> getHyperParametersPermutations (){
			List<HyperParameters> result = new ArrayList<>();
			for(Majority majority: Majority.values()){
				for(PDistanceOptions pDistance: PDistanceOptions.values()){
					for(int k = 1; k <= 20; k++){
						HyperParameters hp = new HyperParameters();
						hp.k = k;
						hp.pDistance = pDistance;
						hp.majority = majority;
						result.add(hp);
					}
				}
			}
			return result;
		}
		
		public String toString(){
			return MessageFormat.format("k: {0}, 1-p Distance: {1}, Majority: {2}", this.k, this.pDistance, this.k);
		}
	}
}
