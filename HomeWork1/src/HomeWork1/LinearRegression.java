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
	}
	
	private void setAlpha(Instances data) throws Exception {
		double bestAlpha = -1;
		double bestSE = Double.MAX_VALUE;
		for(int i = -17; i <= 2; i++){
			m_alpha = Math.pow(3, i);
			m_coefficients = gradientDescent(data);

			double se = calculateSE(data);
			System.out.println("i=" + i + " alpha=" + m_alpha + ", squared error for 20k iterations=" + se);
			if( (! Double.isNaN(se)) && se < bestSE){
				bestSE = se;
				bestAlpha = m_alpha;
			}
			System.out.println("Best alpha: " + bestAlpha);
		}
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
		//Guess some value for [teta_0, teta_1, ... , teta_n]
		m_coefficients = new double[m_truNumAttributes + 1];
		for(int i = 0; i < 20000; i++){
			m_coefficients = updateTetaVector(trainingData, m_coefficients);
		}
		
		return m_coefficients;
	}
	
	private double[] updateTetaVector(Instances trainingData, double[] teta_vecor) {
		double[] updated_teta_vector = new double[teta_vecor.length];
		
		// iterate teta_0, teta_1, ..., teta_n
		for(int tetaIdx = 0; tetaIdx < teta_vecor.length; tetaIdx++){
			updated_teta_vector[tetaIdx] = calculateNewTetaValueForIndex(
					tetaIdx, teta_vecor, trainingData);
		}
		return updated_teta_vector;
	}

	private double calculateNewTetaValueForIndex(
		int tetaIdx, double[] teta_vecor, Instances trainingData) {
		double sum = 0;
		
		// iterate data instances
		for (int i = 0; i < trainingData.numInstances(); i++) { 
			Instance instance = trainingData.instance(i); 
			double sumForInstance = teta_vecor[0];
			
			// iterate attributes
			for (int k = 0; k < m_truNumAttributes + 1; k++) {
				double attributeValue = instance.value(k);
				if(k == m_ClassIndex){
					sumForInstance-= attributeValue;
					continue;
				}
				sumForInstance+= teta_vecor[k + 1] * attributeValue;
			}
			if(tetaIdx != 0){
				sumForInstance*= instance.value(tetaIdx);
			}
			sum+= sumForInstance;
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
		double result = 0;
		for (int k = 0; k < m_truNumAttributes + 1; k++) {
			double attributeValue = instance.value(k);
			if(k == 0){
				result+= m_coefficients[0];
				continue;
			}
			if(k == m_ClassIndex){
				continue;
			}
			result+= m_coefficients[k] * attributeValue;
		}
		return result;
	}
	
	/**
	 * Calculates the total squared error over the data on a linear regression
	 * predictor with weights given by m_coefficients.
     *
	 * @param testData
	 * @return
	 * @throws Exception
	 */
	public double calculateSE(Instances testData) throws Exception {
		double sum = 0;
		for (int i = 0; i < testData.numInstances(); i++) { 
			Instance instance = testData.instance(i); 
			double prediction = regressionPrediction(instance);
			sum+= Math.pow(prediction - instance.value(m_ClassIndex), 2);		
		}
		return (sum / testData.numInstances()) / 2;
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
