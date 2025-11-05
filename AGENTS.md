The current project aims to build a Machine Learning model that will predict the fuel consumption of segments of flights given the trajectory of the flight. The following information is available:
1. Data dictionary and data profile of the data files available along with columns etc. See  @data_profile_report.txt 
2.  @run_pipeline.py  which is used to drive the data preparation process, training. This is missing the prediction component which must be added later. The idea is to enhance the project with a number of ML techniques and compare them. For now only GBR is implemented. 
3.  @data_preparation.py which handles the data preparation. 
4.  @config.py which contains configuration parameters for the whole project.  
5.  @acPerfOpenAP.csv contains aircraft flight parameters 
6.  @train.py  handles the GBR training. The idea is to have different files for different ML techniques. 
7.  @create_submission.py will create the submission file which will use the flightlist_rank.parquet file to fill in the fuel_rank_submission.parquet for final evaluation by the competition authorities. 

Your task is to assist in the completion of the GBR pipeline that will allow to train and test a small sample of the data for testing purposes. Each training session must save all needed information such as the training set and the testing set so that testing can be run following the training easily by selecting the model. 