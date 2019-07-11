from pylab import *
import networkx as nx
import math
from numpy.linalg import inv
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
import copy
from numpy import linalg as LA
import csv
import array
import random
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn import linear_model
from keras.callbacks import ModelCheckpoint

#load csv, ignore the first row,type=int, data read as int， else float
def loadCSV(filename,type): 
        matrix_data=[]
        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            for row_vector in csvreader:
                if type=='int':
                    matrix_data.append(list(map(int,row_vector[1:])))
                else:
                    matrix_data.append(list(map(float,row_vector[1:])))
        return np.matrix(matrix_data)
def fitFunc(individual,parameter1,parameter2):
       label_known=parameter1
       multPred=parameter2
       ensemble_prediction=np.zeros(len(label_known))
       for i in range(0,len(multPred)):
            temp=np.zeros(len(label_known))
            for j in range(len(temp)):
                temp[j]=(individual[i]*multPred[i])[0]
            ensemble_prediction=ensemble_prediction+temp
       precision, recall, pr_thresholds = precisionRecallCurve(label_known, ensemble_prediction)
       auprScore = auc(recall, precision)
       return (auprScore),

class MethodHub():
    def neighborRecoMethod(simMatrix,trainMat):
        #neighbor recommender method
        finalMatrix=np.matrix(trainMat)*np.matrix(simMatrix)
        D=np.diag(((simMatrix.sum(axis=1)).getA1()))
        finalMatrix=finalMatrix*pinv(D)
        finalMatrix=finalMatrix+np.transpose(finalMatrix)
        return finalMatrix

    def labelProp(simMatrix,trainMat):
       #random walk method
       alpha=0.9
       simMatrix=np.matrix(simMatrix)
       trainMat=np.matrix(trainMat)
       D=np.diag(((simMatrix.sum(axis=1)).getA1()))
       N=pinv(D)*simMatrix
       transform_matrix=(1-alpha)*pinv(np.identity(len(simMatrix))-alpha*N)
       finalMatrix=transform_matrix*trainMat
       finalMatrix=finalMatrix+np.transpose(finalMatrix)
       return finalMatrix


#cross validation
def crossValidation(dd_matrix, cvNo, seed):	 
    lnkNo = 0
    lnkPos = []
    nonlnkPos = [] 
    for i in range(0, len(dd_matrix)):
        for j in range(i + 1, len(dd_matrix)):
            if dd_matrix[i, j] == 1:
                lnkNo = lnkNo + 1
                lnkPos.append([i, j])
            else:
                nonlnkPos.append([i, j])
    lnkPos = np.array(lnkPos)
    index = np.arange(0, lnkNo)
    random.shuffle(index)
    foldNo = lnkNo // cvNo
    for cv in range(cvNo):
        r=random.randint(1, lnkNo-550)
        testInd=index[r:r+500]
        testInd.sort()
        testLnkPos = lnkPos[testInd]
        trainMat = copy.deepcopy(dd_matrix)
        for i in range(0, len(testLnkPos)):
            trainMat[testLnkPos[i, 0], testLnkPos[i, 1]] = 0
            trainMat[testLnkPos[i, 1], testLnkPos[i, 0]] = 0
            testPos = list(testLnkPos) + list(nonlnkPos)
        #  GA
        weights,cf1,cf2 = internal_determine_parameter(copy.deepcopy(trainMat))
        [multPredictMat,multPredictRes] = ensemble_method(copy.deepcopy(dd_matrix), trainMat, testPos)
        # logstic weight

        ensembleRes, ensembleRes_cf1,ensembleRes_cf2= ensemble_scoring(copy.deepcopy(dd_matrix), multPredictMat,testPos, weights,cf1,cf2)
        for i in range(0,len(multPredictRes)):
            [aucScore, auprScore, precision, recall, accuracy, f]=multPredictRes[i]
            resFile.write(aucScore+' '+auprScore+' '+precision+' '+recall+' '+accuracy+' '+f+"\n")
            resFile.flush()

        [aucScore, auprScore, precision, recall, accuracy, f] = ensembleRes
        resFile.write(aucScore + ' ' + auprScore + ' ' + precision+ ' ' + recall + ' ' + accuracy + ' ' + f + "\n")
        resFile.flush()

        [aucScore, auprScore, precision, recall, accuracy, f] = ensembleRes_cf1
        resFile.write(aucScore + ' ' + auprScore + ' ' + precision + ' ' + recall + ' ' + accuracy + ' ' + f + "\n")
        resFile.flush()

        [aucScore, auprScore, precision, recall, accuracy, f] = ensembleRes_cf2
        resFile.write(aucScore + ' ' + auprScore + ' ' + precision+ ' ' + recall + ' ' + accuracy + ' ' + f + "\n")
        resFile.flush()

        wtsString=''
        for i in range(0,len(weights)):
            wtsString=wtsString+' '+str(weights[i])
        wtsFile.write(wtsString + "\n")
        resFile.flush()
        wtsFile.flush()


#compute cross validation results
def modelEval(real_matrix,PredictMat,testPos,featureName): 
       label_known=[]
       predictProb=[]

       for i in range(0,len(testPos)):
           label_known.append(real_matrix[testPos[i][0],testPos[i][1]])
           predictProb.append(PredictMat[testPos[i][0],testPos[i][1]])
       normalize=MinMaxScaler()
       predictProb=np.array(predictProb)
       predictProb=predictProb.reshape(-1,1)
       predictProb= normalize.fit_transform(predictProb)
       label_known=np.array(label_known)
       predictProb=np.array(predictProb)

       precision, recall, pr_thresholds = precisionRecallCurve(label_known, predictProb)
       auprScore = auc(recall, precision)

       fMeasure=np.zeros(len(pr_thresholds))
       for k in range(0,len(pr_thresholds)):
           if (precision[k]+precision[k])>0:
              fMeasure[k]=2*precision[k]*recall[k]/(precision[k]+recall[k])
           else:
              fMeasure[k]=0
       maxInd=fMeasure.argmax()
       threshold=pr_thresholds[maxInd]

       fpr, tpr, auc_thresholds = roc_curve(label_known, predictProb)
       aucScore = auc(fpr, tpr)
       predScore=np.zeros(len(label_known))
       for i in range(len(predScore)):
            if(predictProb[i]>threshold):
                predScore[i]=1

       f=f1_score(label_known,predScore)
       accuracy=accuracy_score(label_known,predScore)
       precision=precision_score(label_known,predScore)
       recall=recall_score(label_known,predScore)
       print('results for feature:'+featureName)
       print('AUC score:%.2f, AUPR score:%.2f, recall score:%.2f, precision score:%.2f, accuracy:%.2f, f-measure:%.2f' %(aucScore,auprScore,recall,precision,accuracy,f))
       aucScore, auprScore, precision, recall, accuracy, f = ("%.3f" % aucScore), ("%.3f" % auprScore), ("%.3f" % precision), ("%.3f" % recall), ("%.3f" % accuracy), ("%.3f" % f)
       results=[aucScore,auprScore,precision, recall,accuracy,f]
       return results





def getPara(real_matrix, multMatrix, testPos):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)
    variable_num = len(multMatrix)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, variable_num)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    label_known = []
    for i in range(0, len(testPos)):
        label_known.append(real_matrix[testPos[i][0], testPos[i][1]])

    multPred = []
    for i in range(0, len(multMatrix)):
        predictProb = []
        PredictMat = multMatrix[i]
        for j in range(0, len(testPos)):
            predictProb.append(PredictMat[testPos[j][0], testPos[j][1]])
        normalize = MinMaxScaler()
        predictProb=np.array(predictProb)
        predictProb=predictProb.reshape(-1,1)

        predictProb = normalize.fit_transform(predictProb)
        multPred.append(predictProb)

    toolbox.register("evaluate", fitFunc, parameter1=label_known, parameter2=multPred)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    random.seed(0)
    pop = toolbox.population(n=100) #n=100
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50, stats=stats, halloffame=hof, verbose=True) #ngen=50
    pop.sort(key=lambda ind: ind.fitness, reverse=True)
    return pop[0]


#ensemble method
def ensemble_method(dd_matrix,trainMat,testPos):
    #load feature similarity values for every drug pair
    chem_sim_simMatrix=loadCSV('dataset/chem_Jacarrd_sim.csv','float')
    target_simMatrix=loadCSV('dataset/target_Jacarrd_sim.csv','float')
    transporter_simMatrix=loadCSV('dataset/transporter_Jacarrd_sim.csv','float')
    enzyme_simMatrix=loadCSV('dataset/enzyme_Jacarrd_sim.csv','float')
    pathway_simMatrix=loadCSV('dataset/pathway_Jacarrd_sim.csv','float')
    indication_simMatrix=loadCSV('dataset/indication_Jacarrd_sim.csv','float')
    label_simMatrix=loadCSV('dataset/sideeffect_Jacarrd_sim.csv','float')
    offlabel_simMatrix=loadCSV('dataset/offsideeffect_Jacarrd_sim.csv','float')


    multMatrix=[]
    multiple_result = []

    #Neighbor recommender method on the eight features
    PredictMat=MethodHub.neighborRecoMethod(chem_sim_simMatrix,trainMat)
    results=modelEval(dd_matrix,PredictMat,testPos,'chem_neighbor')
    multiple_result.append(results)
    multMatrix.append(PredictMat)

    PredictMat=MethodHub.neighborRecoMethod(target_simMatrix,trainMat)
    results =modelEval(dd_matrix,PredictMat,testPos,'target_neighbor')
    multiple_result.append(results)
    multMatrix.append(PredictMat)

    PredictMat=MethodHub.neighborRecoMethod(transporter_simMatrix,trainMat)
    results =modelEval(dd_matrix,PredictMat,testPos,'transporter_neighbor')
    multiple_result.append(results)
    multMatrix.append(PredictMat)

    PredictMat=MethodHub.neighborRecoMethod(enzyme_simMatrix,trainMat)
    results =modelEval(dd_matrix,PredictMat,testPos,'enzyme_neighbor')
    multiple_result.append(results)
    multMatrix.append(PredictMat)

    PredictMat=MethodHub.neighborRecoMethod(pathway_simMatrix,trainMat)
    results =modelEval(dd_matrix,PredictMat,testPos,'pathway_neighbor')
    multiple_result.append(results)
    multMatrix.append(PredictMat)

    PredictMat=MethodHub.neighborRecoMethod(indication_simMatrix,trainMat)
    results =modelEval(dd_matrix,PredictMat,testPos,'indication_neighbor')
    multiple_result.append(results)
    multMatrix.append(PredictMat)

    PredictMat=MethodHub.neighborRecoMethod(label_simMatrix,trainMat)
    results =modelEval(dd_matrix,PredictMat,testPos,'label_neighbor')
    multiple_result.append(results)
    results =multMatrix.append(PredictMat)

    PredictMat=MethodHub.neighborRecoMethod(offlabel_simMatrix,trainMat)
    results =modelEval(dd_matrix,PredictMat,testPos,'offlabel_neighbor')
    multiple_result.append(results)
    multMatrix.append(PredictMat)

    
    #Random walk method on the eight features
    PredictMat=MethodHub.labelProp(chem_sim_simMatrix,trainMat)
    results =modelEval(dd_matrix,PredictMat,testPos,'chem_label')
    multiple_result.append(results)
    multMatrix.append(PredictMat)  

    PredictMat=MethodHub.labelProp(target_simMatrix,trainMat)
    results =modelEval(dd_matrix,PredictMat,testPos,'target_label')
    multiple_result.append(results)
    multMatrix.append(PredictMat)

    PredictMat=MethodHub.labelProp(transporter_simMatrix,trainMat)
    results =modelEval(dd_matrix,PredictMat,testPos,'transporter_label')
    multiple_result.append(results)
    multMatrix.append(PredictMat)

    PredictMat=MethodHub.labelProp(enzyme_simMatrix,trainMat)
    results =modelEval(dd_matrix,PredictMat,testPos,'enzyme_label')
    multiple_result.append(results)
    multMatrix.append(PredictMat)

    PredictMat=MethodHub.labelProp(pathway_simMatrix,trainMat)
    results =modelEval(dd_matrix,PredictMat,testPos,'pathway_label')
    multiple_result.append(results)
    multMatrix.append(PredictMat)

    PredictMat=MethodHub.labelProp(indication_simMatrix,trainMat)
    results =modelEval(dd_matrix,PredictMat,testPos,'indication_label')
    multiple_result.append(results)
    multMatrix.append(PredictMat)

    PredictMat=MethodHub.labelProp(label_simMatrix,trainMat)
    results =modelEval(dd_matrix,PredictMat,testPos,'label_label')
    multiple_result.append(results)
    multMatrix.append(PredictMat)

    PredictMat=MethodHub.labelProp(offlabel_simMatrix,trainMat)
    results =modelEval(dd_matrix,PredictMat,testPos,'offlabel_label')
    multiple_result.append(results)
    multMatrix.append(PredictMat)   

    return multMatrix,multiple_result

def internal_determine_parameter(dd_matrix):
    trainMat,testPos=holdout_by_link(copy.deepcopy(dd_matrix),0.2,1)
    [multMatrix,multiple_result]=ensemble_method(copy.deepcopy(dd_matrix),trainMat,testPos)
    weights=getPara(copy.deepcopy(dd_matrix),multMatrix,testPos)
    input_matrix=[]
    output_matrix = []
    for i in range(0, len(testPos)):
        vector=[]
        for j in range(0, len(multMatrix)):
           vector.append(multMatrix[j][testPos[i][0], testPos[i][1]])
        input_matrix.append(vector)
        output_matrix.append(dd_matrix[testPos[i][0], testPos[i][1]])
    input_matrix=np.array(input_matrix)
    output_matrix= np.array(output_matrix)
    clf1 = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)   
    if (np.sum(output_matrix)) in [len(output_matrix),0]:
        print ("all one class")
    else:
        clf1.fit(input_matrix, output_matrix)
    
    clf2 = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
    if (np.sum(output_matrix)) in [len(output_matrix),0]:
        print ("all one class")
    else:
        clf2.fit(input_matrix, output_matrix)
    return weights,clf1, clf2


def holdout_by_link(dd_matrix, ratio, seed):
    lnkNo = 0
    lnkPos = []
    nonlnkPos = []  # all non-link position
    for i in range(0, len(dd_matrix)):
        for j in range(i + 1, len(dd_matrix)):
            if dd_matrix[i, j] == 1:
                lnkNo = lnkNo + 1
                lnkPos.append([i, j])
            else:
                nonlnkPos.append([i, j])

    lnkPos = np.array(lnkPos)
    random.seed(seed)
    index = np.arange(0, lnkNo)
    random.shuffle(index)
    trainInd = index[(int(lnkNo * ratio) + 1):]
    testInd = index[0:int(lnkNo * ratio)]
    trainInd.sort()
    testInd.sort()
    testLnkPos = lnkPos[testInd]
    trainMat = copy.deepcopy(dd_matrix)

    for i in range(0, len(testLnkPos)):
        trainMat[testLnkPos[i, 0], testLnkPos[i, 1]] = 0
        trainMat[testLnkPos[i, 1], testLnkPos[i, 0]] = 0
    testPos = list(testLnkPos) + list(nonlnkPos)
    return trainMat, testPos


def ensemble_scoring(real_matrix, multMatrix, testPos, weights,cf1,cf2):
    label_known = []
    for i in range(0, len(testPos)):
        label_known.append(real_matrix[testPos[i][0], testPos[i][1]])

    multPred = []
    for i in range(0, len(multMatrix)):
        predictProb = []
        PredictMat = multMatrix[i]
        for j in range(0, len(testPos)):
            predictProb.append(PredictMat[testPos[j][0], testPos[j][1]])
        normalize = MinMaxScaler()
        predictProb=np.array(predictProb)
        predictProb=predictProb.reshape(-1,1)

        predictProb = normalize.fit_transform(predictProb)
        predictProb=np.array(predictProb)
        multPred.append(predictProb)
    ensemble_prediction = np.zeros(len(label_known))
    for i in range(0,len(multPred)):
        temp=np.zeros(len(label_known))
        for j in range(len(temp)):
            temp[j]=(weights[i]*multPred[i])[0]
        ensemble_prediction=ensemble_prediction+temp

    ensemble_prediction_cf1 = np.zeros(len(label_known))
    ensemble_prediction_cf2= np.zeros(len(label_known))
    for i in range(0, len(testPos)):
        vector=[]
        for j in range(0, len(multMatrix)):
           vector.append(multMatrix[j][testPos[i][0], testPos[i][1]])
        vector=np.array(vector)
        vector=vector.reshape(-1,8)
        aa=cf1.predict_proba(vector)
        ensemble_prediction_cf1[i]=(cf1.predict_proba(vector))[0][1]
        ensemble_prediction_cf2[i]=(cf2.predict_proba(vector))[0][1]

    normalize = MinMaxScaler()
    predictProb=np.array(predictProb)
    predictProb=predictProb.reshape(-1,1)
    ensemble_prediction=ensemble_prediction.reshape(-1,1)
    ensemble_prediction = normalize.fit_transform(ensemble_prediction)

    result = calculate_metric_score(label_known, ensemble_prediction)
    result_cf1=calculate_metric_score(label_known, ensemble_prediction_cf1)
    result_cf2=calculate_metric_score(label_known, ensemble_prediction_cf2)
    return result,result_cf1,result_cf2

def calculate_metric_score(label_known,predict_score):
   precision, recall, pr_thresholds = precisionRecallCurve(label_known, predict_score)
   auprScore = auc(recall, precision)

   fMeasure = np.zeros(len(pr_thresholds))
   for k in range(0, len(pr_thresholds)):
      if (precision[k] + precision[k]) > 0:
          fMeasure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
      else:
          fMeasure[k] = 0
   maxInd = fMeasure.argmax()
   threshold = pr_thresholds[maxInd]
   fpr, tpr, auc_thresholds = roc_curve(label_known, predict_score)
   aucScore = auc(fpr, tpr)

   predScore=np.zeros(len(label_known))
   for i in range(len(predScore)):
       if(predScore[i]>threshold):
               predScore[i]=1

   f = f1_score(label_known, predScore)
   accuracy = accuracy_score(label_known, predScore)
   precision = precision_score(label_known, predScore)
   recall = recall_score(label_known, predScore)
   print('results for feature:' + 'weighted_scoring')
   print('AUC score:%.2f, AUPR score:%.2f, recall score:%.2f, precision score:%.2f, accuracy:%.2f' % (aucScore, auprScore, recall, precision, accuracy))
   aucScore, auprScore, precision, recall, accuracy, f = ("%.3f" % aucScore), ("%.3f" % auprScore), ("%.3f" % precision), ("%.3f" % recall), ("%.3f" % accuracy), ("%.3f" % f)
   results = [aucScore, auprScore, precision, recall, accuracy, f]
   return results


runtimes=20  #runtimes =20
dd_matrix = loadCSV('dataset/drug_drug_matrix.csv', 'int') #load dataset
resFile_str="result/result_on_our_dataset_5CV"
weights_results_str="result/weights_on_our_dataset_5CV"
for seed in range(0, runtimes):
    resFile_path=resFile_str+"_"+str(seed)+".txt"
    weights_results_path=weights_results_str+"_"+str(seed)+".txt"
    resFile = open(resFile_path, "w")
    wtsFile = open(weights_results_path, "w")
    crossValidation(dd_matrix,5, seed) 