package HomeWork1;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class LinearRegression implements Classifier {
	private static final double EPSILON = 0.003;
	
	
    private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha;
	
	//the method which runs to train the linear regression predictor, i.e.
	//finds its weights.
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		trainingData = new Instances(trainingData);
		m_ClassIndex = trainingData.classIndex();
		//since class attribute is also an attribute we subtract 1
		m_truNumAttributes = trainingData.numAttributes() - 1;

		//Guess some value for [teta_0, teta_1, ... , teta_n]
		m_coefficients = new double[m_truNumAttributes + 1];

		// Init thetas to random values
		for (int i = 0; i < m_coefficients.length; i++)
			m_coefficients[i] = Math.random();

		setAlpha(trainingData);

		double cur_error = Double.MAX_VALUE;
		System.out.println("Using alpha: " + m_alpha);
		boolean found_thetas = false;
		int count = 1;
		while (!found_thetas){
			System.out.println("Iteration " + count++);
			for(int i = 0; i < 100; i++){
				m_coefficients = updateTetaVector(trainingData, m_coefficients);
			double tempError = calculateSE(trainingData);
			if (cur_error -  tempError > EPSILON)
				cur_error = tempError;
			else
				found_thetas = true;
			}
		}
		System.out.println("Found correct thetas");
	}
	
	private void setAlpha(Instances data) throws Exception {
		double bestAlpha = -1;
		double bestSE = Double.MAX_VALUE;
		double tempSE;
		for(int i = -17; i <= 2; i++){
			m_alpha = Math.pow(3, i);

			for(int j = 0; j < 20000; j++)
				m_coefficients = gradientDescent(data);
			tempSE = calculateSE(data);
			System.out.println("i=" + i + " alpha=" + m_alpha + ", squared error for 20k iterations=" + tempSE);
			if (Double.isNaN(tempSE))
				tempSE = Double.MAX_VALUE;

			if(tempSE < bestSE){
				bestSE = tempSE;
				bestAlpha = m_alpha;
			}
		}
		System.out.println("Best alpha: " + bestAlpha);
		m_alpha = bestAlpha;
	}
	
	/**
	 * An implementation of the gradient descent algorithm which should
	 * return the weights of a linear regression predictor which minimizes
	 * the average squared error.
     * 
	 * @param trainingData
	 * @throws Exception
	 */
	private double[] gradientDescent(Instances trainingData) throws Exception {
		m_coefficients = updateTetaVector(trainingData, m_coefficients);

		return m_coefficients;
	}
	
	private double[] updateTetaVector(Instances trainingData, double[] teta_vecor) {
		double[] updated_teta_vector = new double[teta_vecor.length];
		
		// iterate teta_0, teta_1, ..., teta_n
		for(int tetaIdx = 0; tetaIdx < teta_vecor.length; tetaIdx++)
			updated_teta_vector[tetaIdx] = calculateNewTetaValueForIndex(tetaIdx, teta_vecor, trainingData);
		return updated_teta_vector;
	}

	private double calculateNewTetaValueForIndex(
		int tetaIdx, double[] teta_vecor, Instances trainingData) {
		double sum = 0;

		// iterate data instances
		for (int i = 0; i < trainingData.numInstances(); i++) { 
			Instance instance = trainingData.instance(i);

			// No need to multiply first theta
			double sumForInstance = teta_vecor[0];

			// iterate attributes
			for (int k = 1; k < m_truNumAttributes + 1; k++) {
				double attributeValue = instance.value(k);

				// Add theta_n * x_n to the sum
				sumForInstance+= teta_vecor[k] * attributeValue;

				// Subtract the true output from the sum
				if(k == m_ClassIndex){
					sumForInstance -= attributeValue;
					continue;
				}

			}
			// For each theta (except theta_0) multiple the sum by x_n
			if(tetaIdx != 0)
				sumForInstance *= instance.value(tetaIdx);

			sum += sumForInstance;
		}
		return teta_vecor[tetaIdx] - (m_alpha * sum / trainingData.numInstances());
	}

	/**
	 * Returns the prediction of a linear regression predictor with weights
	 * given by m_coefficients on a single instance.
     *
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public double regressionPrediction(Instance instance) throws Exception {
		// No need to multiply the first theta
		double result = m_coefficients[0];
		for (int k = 1; k < m_truNumAttributes + 1; k++) {
			if(k == m_ClassIndex)
				continue;

			// Multiply theta_n with appropriate x_n and add to result
			result+= m_coefficients[k] * instance.value(k);
		}
		return result;
	}
	
	/**
	 * Calculates the total squared error over the data on a linear regression
	 * predictor with weights given by m_coefficients.
     *
	 * @param data
	 * @return
	 * @throws Exception
	 */
	public double calculateSE(Instances data) throws Exception {
		double sum = 0;
		for (int i = 0; i < data.numInstances(); i++) {
			Instance instance = data.instance(i);
			double prediction = regressionPrediction(instance);
			sum+= Math.pow(prediction - instance.value(m_ClassIndex), 2);		
		}
		return sum / (2.0 * (data.numInstances()));
	}
    
    @Override
	public double classifyInstance(Instance arg0) throws Exception {
		// Don't change
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
