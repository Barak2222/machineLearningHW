package HomeWork7;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;


import weka.core.Instance;
import weka.core.Instances;

public class KMeans {
	private int m_k;
	private Instances m_centroids;
	private boolean m_printErrorInEachIteration;
	private final int PRESET_NUMBER_OF_IERATIONS = 40;
	private HashMap<Instance, Instances> map;
	
	/**
	 * This method is building the KMeans object. It should initialize centroids (by calling initializeCentroids)
	 * and run the K-Means algorithm (which means to call findKMeansCentroids methods).
	 * @param instances
	 */
	public void buildClusterModel(Instances instances, boolean printErrorInEachIteration) {
		m_centroids = new Instances(instances);
		m_centroids.clear();
		initializeCentroids(instances);
		m_printErrorInEachIteration = printErrorInEachIteration;
		findKMeansCentroids(instances);
	}

	public void setK(int k){
		m_k = k;
	}
	
	/**
	 * Initialize the centroids by selecting k random instances from the training set and setting the centroids to be those instances.
	 * @param instances
	 */
	private void initializeCentroids(Instances instances) {
		if(instances.size() < m_k){
			throw new IllegalArgumentException();
		}
		Set<Instance> randomInstances = new HashSet<>();
		while(randomInstances.size() < m_k){
			int rand = (int) (m_k * Math.random());
			randomInstances.add(instances.get(rand));
		}
		m_centroids.addAll(randomInstances);
	}
	
	private void resetCentroidsMap(Instances emptyInstances){
		int i = 0;
		for (Instance centroid : m_centroids) {
			System.out.println("resetCentroidsMap: Iteration #" + (i++));
			map.put(centroid, new Instances(emptyInstances));
		}
	}

	/**
	 * Should find and store the centroids according to the KMeans algorithm. Your stopping condition for when to stop
	 * iterating can be either when the centroids have not moved much from their previous location, 
	 * the cost function did not change much, or you have reached a preset number of iterations. In this assignment we 
	 * will only use the preset number option. A good preset number of iterations is 40. Use one or any combination of these 
	 * methods to determine when to stop iterating.
	 * @param instances
	 */
	private void findKMeansCentroids(Instances instances) {
		Instances emptyInstances = new Instances(instances);
		emptyInstances.clear();
		Instances newCentroids;
		
		for (int i = 0; i < PRESET_NUMBER_OF_IERATIONS; i++) {
			newCentroids = new Instances(m_centroids);
			newCentroids.clear();
			System.out.println("findKMeansCentroids: Iteration #" + i);
			resetCentroidsMap(emptyInstances);
			System.out.println("Resetted centroid map");
			
			// Assign each instance to their closest centroid8
			for (Instance instance : instances)
				map.get(m_centroids.get(findClosestCentroid(instance))).add(instance);
			
			for (Instance centroid : map.keySet()) {
				Instance newCentroid = (Instance) centroid.copy();
				// Clear new centroid attributes values
				for (int j = 0; j < newCentroid.numAttributes(); j++)
					newCentroid.setValue(j, 0);
				
				// Sum up all attributes
				for (Instance instance : map.get(centroid)) {
					for (int j = 0; j < instance.numAttributes(); j++){
						newCentroid.setValue(j, newCentroid.value(j) + instance.value(j));
					}
				}
				// Divide by number of instances
				for (int j = 0; j < newCentroid.numAttributes(); j++)
					newCentroid.setValue(j, newCentroid.value(j) / map.get(centroid).size());
				
				// Add new centroid to a temporary centroids list
				newCentroids.add(newCentroid);
				
			}
			m_centroids = newCentroids;
			
		}
	}


	/**
	 *  Input: 2 Instance objects – one is an instance from the dataset and one is a centroid (if you're using different 
	 *  data structure for the centroid, feel free to change the input). Output: should calculate the squared distance between the 
	 *  input instance and the input centroid.
	 * @param a an instance from the dataset
	 * @param b a centroid
	 * @return should calculate the squared distance between the input instance and the input centroid.
	 */
	private double calcSquaredDistanceFromCentroid(Instance a, Instance b) {
		double squaredDistance = 0;
		for (int i = 0; i < a.numAttributes(); i++) {
			squaredDistance += Math.pow(a.value(i), b.value(i));
		}
		squaredDistance = Math.sqrt(squaredDistance);
		
		return squaredDistance;
	}

	/**
	 * return the index of the closest centroid to the input instance
	 * @param instance
	 * @return the index of the closest centroid to the input instance
	 */
	private int findClosestCentroid(Instance instance) {
		double minDistance = calcSquaredDistanceFromCentroid(instance, m_centroids.get(0));
		int closestIdx = 0;
		double tmpDistance;
		for (int i = 1; i < m_centroids.size(); i++) {
			tmpDistance = calcSquaredDistanceFromCentroid(instance, m_centroids.get(i));
			if (tmpDistance < minDistance){ 
				minDistance = tmpDistance;
				closestIdx = i;
			}
		}
		return closestIdx;
	}

	/**
	 *  Output: should replace every instance in Instances by the centroid to which it is assigned 
	 *  (closest centroid) and return the new Instances object.
	 * @param instances
	 * @return should replace every instance in Instances by the centroid to which it is assigned (closest centroid) and return the new Instances object.
	 */
	public Instances quantize(Instances instances) {
		for(int i = 0; i < instances.size(); i++){
			instances.set(i, m_centroids.get(findClosestCentroid(instances.get(i))));
		}
		return instances;
	}

	/**
	 * Calculate the average within set sum of squared errors. That is it should calculate the average squared distance of every
	 * instance from the centroid to which it is assigned. This is Tr(Sc) from class, divided by the number of instances. 
	 * @param instances
	 * @return the double value of the WSSSE.
	 */
	public double calcAvgWSSSE(Instances instances) {
		double sum = 0;
		for (Instance instance : instances) {
			Instance centroid = m_centroids.get(findClosestCentroid(instance));
			sum+= calcSquaredDistanceFromCentroid(centroid, instance);
		}
		return sum/= instances.size();
	}

}
