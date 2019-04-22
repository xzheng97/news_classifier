import java.util.HashMap;
import java.util.Map;
import java.lang.Math;

/**
 * An implementation of a naive bayes classifier. 
 */

public class NaiveBayesClassifierImpl implements NaiveBayesClassifier {
	private Instance[] m_trainingData;
	private int m_v;
	private double m_delta;
	public int m_sports_count, m_business_count;
	public int m_sports_word_count, m_business_word_count;
	private HashMap<String,Integer> m_map[] = new HashMap[2];

  /**
   * Trains the classifier with the provided training data and vocabulary size
   */
  @Override
  public void train(Instance[] trainingData, int v) {
   
  	  m_trainingData = trainingData;
  	  m_v = v;
  	  m_map[0] = new HashMap<>();
  	  m_map[1] = new HashMap<>();
  	  
  	  // For all the words in the documents, count the number of occurrences. Save in HashMap  
  	  for (Instance ins: m_trainingData) {
  		  // create a hashmap to store instances with label SPORTS
  		  if (ins.label.equals(Label.SPORTS)) {
  			  for (String word: ins.words) {
  				  // case when the word has already appeared, update the count
  				  if (m_map[0].containsKey(word)) 
  					  m_map[0].replace(word, m_map[0].get(word) + 1);
  				  // case when the word appears the first time
  				  else
  					  m_map[0].put(word, 1);
  			  }
  		  }
  		  // create another hashmap to store instances with label BUSINESS
  		  else if (ins.label.equals(Label.BUSINESS)){
  			 for (String word: ins.words) {
 				  if (m_map[1].containsKey(word)) 
 					  m_map[1].replace(word, m_map[1].get(word) + 1);
 				  else
 					  m_map[1].put(word, 1);
 			  }
  		  }
  	  }
  	  
  	  
  }

  /*
   * Counts the number of documents for each label
   */
  public void documents_per_label_count(Instance[] trainingData){
    m_sports_count = 0;
    m_business_count = 0;
    // traverse the training data to update the count of documents with different labels
    for (Instance ins: trainingData) {
    	if (ins.label.equals(Label.SPORTS))
    		m_sports_count++;
    	else if (ins.label.equals(Label.BUSINESS))
    		m_business_count++;
    }
    
  }

  /*
   * Prints the number of documents for each label
   */
  public void print_documents_per_label_count(){
  	  System.out.println("SPORTS=" + m_sports_count);
  	  System.out.println("BUSINESS=" + m_business_count);
  }


  /*
   * Counts the total number of words for each label
   */
  public void words_per_label_count(Instance[] trainingData){
    m_sports_word_count = 0;
    m_business_word_count = 0;
    // traverse the training data
    for (Instance ins: trainingData) {
    	if (ins.label.equals(Label.SPORTS)) 
    		// update the word count of documents with label SPORTS
    		m_sports_word_count += ins.words.length;
    	else if (ins.label.equals(Label.BUSINESS))
    		// update the word count of documents with label BUSINESS
    		m_business_word_count += ins.words.length;
    }
  }

  /*
   * Prints out the number of words for each label
   */
  public void print_words_per_label_count(){
  	  System.out.println("SPORTS=" + m_sports_word_count);
  	  System.out.println("BUSINESS=" + m_business_word_count);
  }

  /**
   * Returns the prior probability of the label parameter, i.e. P(SPORTS) or P(BUSINESS)
   */
  @Override
  public double p_l(Label label) {
    // Calculate the probability for the label. No smoothing here.
    // Just the number of label counts divided by the number of documents.
    double ret = 0;
    // traverse the training data to update the field variables first
    this.documents_per_label_count(m_trainingData);
    // calculate probability of each label 
    if (label.equals(label.SPORTS)) {
    	ret = (double)m_sports_count / (double)m_trainingData.length;
    }
    else if (label.equals(label.BUSINESS))
    	ret = (double)m_business_count / (double)m_trainingData.length;
    return ret;
  }

  /**
   * Returns the smoothed conditional probability of the word given the label, i.e. P(word|SPORTS) or
   * P(word|BUSINESS)
   */
  @Override
  public double p_w_given_l(String word, Label label) {

    double ret = 0;
    m_delta = 0.00001;
    int numCount = 0;
    // traverse the training data to update the field variables first
    this.words_per_label_count(m_trainingData);
    // Calculate the probability with Laplace smoothing for word in each label
    if (label.equals(label.SPORTS)) {
    	// update numCount only when the word exists
    	if(m_map[0].containsKey(word))
    		numCount = m_map[0].get(word);
    	// Laplace smoothing
    	ret = (numCount + m_delta) / (double)((Math.abs(m_v) * m_delta) + m_sports_word_count);
    }
    // same for the other label
    else if (label.equals(label.BUSINESS)) {
    	if(m_map[1].containsKey(word))
    		numCount = m_map[1].get(word);
    	ret = (numCount + m_delta) / (double)((Math.abs(m_v) * m_delta) + m_business_word_count);
    }
    	
    return ret;
  }

  /**
   * Classifies an array of words as either SPORTS or BUSINESS.
   */
  @Override
  public ClassifyResult classify(String[] words) {
    // Sum up the log probabilities for each word in the input data, and the probability of the label
    // Set the label to the class with larger log probability
    ClassifyResult ret = new ClassifyResult();
    ret.label = Label.SPORTS;
    ret.log_prob_sports = 0;
    ret.log_prob_business = 0;
    
    double sumsports = 0.0;
    double sumbusiness = 0.0;
    // sum up the log of the conditional probability of each words by each label
    for (String word: words) {
    	sumsports += Math.log(p_w_given_l(word, Label.SPORTS));
    	sumbusiness += Math.log(p_w_given_l(word, Label.BUSINESS));
    }
    // plus the log of the probability
    ret.log_prob_sports = Math.log(p_l(Label.SPORTS)) + sumsports;
    ret.log_prob_business = Math.log(p_l(Label.BUSINESS)) + sumbusiness;
    
    if(ret.log_prob_business > ret.log_prob_sports)
    	ret.label = Label.BUSINESS;
    
    
    return ret; 
  }
  
  /*
   * Constructs the confusion matrix
   */
  @Override
  public ConfusionMatrix calculate_confusion_matrix(Instance[] testData){

    int TP, FP, FN, TN;
    TP = 0;
    FP = 0;
    FN = 0;
    TN = 0;
    // Count the true positives, true negatives, false positives, false negatives
    // traverse the test data
    for (Instance ins: testData) {
    	//update each case
    	if (classify(ins.words).label.equals(Label.SPORTS)) {
    		if (ins.label.equals(Label.SPORTS))
    			TP++;
    		else if (ins.label.equals(Label.BUSINESS))
    			FP++;	
    	}
    	
    	else if (classify(ins.words).label.equals(Label.BUSINESS)) {
    		if (ins.label.equals(Label.SPORTS))
    			FN++;
    		else if (ins.label.equals(Label.BUSINESS))
    			TN++;
    	}
    			
    }
    
    return new ConfusionMatrix(TP,FP,FN,TN);
  }
  
}
