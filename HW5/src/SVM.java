package HomeWork5;

import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.core.Instances;

public class SVM {
	public SMO m_smo;

	public SVM() {
		this.m_smo = new SMO();
	}
	
	public void buildClassifier(Instances instances) throws Exception{
		m_smo.buildClassifier(instances);
	}
	
	public int[] calcConfusion(Instances instances) throws Exception{
		
	}
}
