import BrukerMRI as bruker

MainDir = "/Volumes/pr_ohlendorf/fMRI/RawData/Project_MMP9_NJ_MP/Test_animals/20250220_122734_RGRO_250220_0122_RN_SD_010_0122_1_1/"
ExpNum = 9
Experiment = bruker.ReadExperiment(MainDir, ExpNum)

print(Experiment.method["PVM_EchoTime"])
