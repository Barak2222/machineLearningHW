package HomeWork5;

import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.core.Instance;
import weka.core.Instances;

public class SVM {
	public SMO m_smo;
	public Kernel m_kernel;
	public double m_C;

	public SVM() {
		this.m_smo = new SMO();
	}

	public void buildClassifier(Instances instances) throws Exception {
		m_smo.setKernel(m_kernel);
		m_smo.buildClassifier(instances);
	}

	/**
	 * Setting the Weka SMO classifier kernel
	 * 
	 * @param newKernel
	 *            - Kernel object
	 * @throws Exception
	 */
	public void setKernel(Kernel newKernel) throws Exception {
		m_kernel = newKernel;
	}

	/**
	 * Setting the C value for the Weka SMO classifier
	 * 
	 * @param newC
	 *            - double
	 * @throws Exception
	 */
	public void setC(double newC) throws Exception {
		m_C = newC;
		m_smo.setC(newC);
	}

	/**
	 * Getting the C value for the Weka SMO classifier
	 * 
	 * @return double
	 * @throws Exception
	 */
	public double getC() throws Exception {
		return m_C;
	}

	/**
	 * Calculate the TP, FP, TN, FN for a given instances object
	 * recurrence-events is the 0.0 class and will be the NEGATIVE class
	 * no-recurrence-events is the 1.0 class and will be the POSITIVE class
	 * 
	 * @param Instances
	 *            object
	 * @return int array of size 4 in this order [TP, FP, TN, FN]
	 * @throws Exception
	 */
	public int[] calcConfusion(Instances instances) throws Exception {
		int confusionArray[] = { 0, 0, 0, 0 };
		for (Instance instance : instances) {
			double predictedClass = m_smo.classifyInstance(instance);
			double actualClass = instance.value(instance.classIndex());

			// Predicted positive
			if (predictedClass == 0.0) {
				if (actualClass == 0.0)
					confusionArray[0]++; // TP
				else
					confusionArray[1]++; // FP
			} else {// Predicted negative
				if (actualClass == 1.0)
					confusionArray[2]++; // TN
				else
					confusionArray[3]++; // FN
			}
		}
		return confusionArray;
	}
}
