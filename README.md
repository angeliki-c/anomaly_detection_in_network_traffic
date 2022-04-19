## Anomaly Detection

### Anomaly Detection in the Network Traffic

 
Techniques followed
	
	K-Means Clustering
		
		Reveals encoded information within the features of the dataset. Though the states of the encoding
		hyperparameter k is prespecified and issued in the begining to the algorithm. 
		
		It doesn't necessarilly converge to a correct clustering of the examples. There is always the
		possibility that it will settle to some locally optimal solution.
		
		The result of the clustering is not that interpretable.
		
		This method is not reliable in high-dimensional worlds.
		
		It exhibits parallelism and it is preferred in big data frameworks as it may be computed 
		efficiently.

		In this use case we use a variant off KMeans, K-means||, implemented by Spark, which doesn't 
		assume random choice of the initial centroids when the algorithm starts, but follows some criterions 
		in choosing appropriately the initial values of the centroids for improving the quality of clustering.
		
 		Other unsupervised techniques (gaussian mixtures, lda etc) that aim at giving further structure to
	  	the input dataset and highlight what might not constitute a regular (normal) set of input features 
		can also be used. An extensive list of clustering techniques can be found at [1]

  
Data set
	
	The data set [2] that is going to be used in this analysis was built in the frame of the Third Interna-
	tional Knowledge Discovery and Data Mining Tools Competition, which was held in conjunction with KDD-99
	The Fifth International Conference on Knowledge Discovery and Data Mining.
	It consists of approximately 4.9 million connections over the Internet and 42 features, numerical and 
	categorical, giving clues regarding each connection.
	Although in some settings, configuring the cluster on standalone mode and with as many workers as the 
	cores of the machine, along with repartioning appropriatly the data, the data analysis of the 4.9 million
	x 42 data set may be feasible locally, there is a smaller version of the data set, in any case, composed 
	of 494021 records. 
	

Baseline

	1. lda (Latent Dirichlet Analysis) is used for comparison on clustering 's consistency and classification
	   capability
	2. lda and mlp are used for comparing performance at the multiclasification task  
	
	
	
Challenges
	
	Anomaly detection by its nature belongs to a category of problems, where we do not know beforehand what 
	defines an 'anomaly' in a domain and don't have examples for how each and every possible anomaly may look
	like. Therefore, the supervised learning methods in this problem category are not essentially useful in the
	long term.
	Anomaly detection quite often finds application to fraud detection, network intrusion detection, servers or
	sensor equipment's failure discovery and to others areas, yet quite often what may constitute an anomaly 
	compared to the data seen thus far, may not be actually be a fraud or network intrusion, for example, or it
	may trully be a new type of fraud or network threat.
    


Training process

	A 0.9 train and 0.1 test split has been first applied. 
	Secondly, hyperparameter tuning has been performed on a dev set, composed of 0.1 of the train examples and 
	using the Silhouette metric as an indicative performance metric in clustering. The hyperparameters that we
	have chosen to tune are : k, the distance measure used in KMeans and the maximum number of iterations that
	the algorithm should be run, until reaching convergence, for each choice of k. 
	Training of the model follows, using the best parameters emerged from the previous phase.
	

Evaluation

	1. We evaluate the efficacy of KMeans in clustering examples consistently (using the Silhouette metric)
	   against lda.
	2. The effectiveness of clustering in discerning between the different groups of the data, first between
	   normal and anomalous and then between all the different classes of the training data. The results are 
	   compared to those produced by the application of lda as a baseline approach. The metric used is f1-score.
	3. In additition we investigate the classification capability of K-Means on previously unseen data and compare
	   against lda and a mlp classiffier, using F1-score as performance metric.	
	

Performance Metrics

	The Silhouette metric has been used in the evaluation, during hyperparameter tuning, which takes values 
	within [-1, 1], with value 1 representing the situation where the examples belonging to each cluster present 
	small distance or high affinity between each other, and the greatest distance from the examples of all the 
	other clusters.
	
	The same metric has been applied on the test set for evaluating the quality of the clustering on unseen data.
	
	areaUnderROC metric is applied to the training and the test set in order to evaluate the classification ability 
	of the models between the two gross classes of connection type, normal and not normal (anomally detection).
	
	F1-score has also been applied on the test set for evaluating the predictive ability of the model on specific
	network intrusions' detection (multiclassification task).
    

Code

   kmeans_clustering.py
   
   evaluation.py  
   Though the best clustering performed happens for k =2, after conducting Hyperparameter tuning, for the 
   evaluation we pick k = 23, which results in a qualitative clustering as well. The reason for this is
   that after analytical overview of the results on the data clustered, it leads to more meaningful clustering
   of the data. This clustering is not expected to act as classification to the classes of the dataset. Supervised
   learning techniques are designed to be driven to meet this goal.
   
   All can be run interactively with pyspark shell or by submitting e.g. exec(open("project/location/anomaly_detection_in_network_traffic/kmeans_clustering.py").read()) 
   for an all at once execution. The code has been tested on a Spark standalone cluster. For the Spark setting,
   spark-3.1.2-bin-hadoop2.7 bundle has been used.
   The external python packages that are used in this implementation exist in the requirements.txt file. Install with: 
	   pip install -r project/location/anomaly_detection_in_network_traffic/requirements.txt
   This use case is inspired from the series of experiments presented in [3], though it deviates from it, in the
   programming language, the setting used and in the analysis followed.

   

References

	1. https://scikit-learn.org/stable/modules/clustering.html#overview-of-clustering-methods
	2. http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
  	3. Advanced Analytics with Spark, Sandy Ryza, Uri Laserson, Sean Owen, & Josh Wills
 
 
