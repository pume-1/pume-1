from operator import le, truediv
import pickle
import os
import json
import hashlib
from turtle import position
from sklearn.ensemble import IsolationForest
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import joblib 
import seaborn as sns
import random
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
import warnings
import csv
warnings.filterwarnings("ignore")


def get_file_md5(f):
    m = hashlib.md5()
    while True:
        data = f.read(1024).encode("utf-8")  #将文件分块读取
        if not data:
            break
        m.update(data)
    return m.hexdigest()

def novul_hash_dict_get(novul_hash_txt, novul_hash_dict_path):
    if not os.path.exists(novul_hash_dict_path):
        novul_hash_dict = dict()
        with open(novul_hash_txt, 'r') as f:
            novul_path_list = f.readlines()
        for _novul_path in novul_path_list:
            _novul_path = _novul_path.strip()
            slice_list = os.listdir(_novul_path)
            for _slice in slice_list:
                _slice_path = os.path.join(_novul_path, _slice)
                with open(_slice_path, 'r') as f:
                    slice_md5 = get_file_md5(f)
                novul_hash_dict[_slice_path] = slice_md5
        with open(novul_hash_dict_path, 'w+') as f:
            json.dump(novul_hash_dict, f)
    else:
        with open(novul_hash_dict_path, 'r') as f:
            novul_hash_dict = json.load(f)
    return novul_hash_dict

                    
def hash_filter(novul_hash_dict, vul_test_file, filter_res_path):
    if not os.path.exists(filter_res_path):
        keep_slice_dict = dict()
        all_cnt = 0
        keep_cnt = 0
        with open(vul_test_file, 'r') as f:
            vul_list = f.readlines()
        for _vul_path in vul_list:
            _vul_path = _vul_path.strip()
            slice_list = os.listdir(_vul_path)
            keep_slice_dict[_vul_path] = []
            for _slice in slice_list:
                _slice_path = os.path.join(_vul_path, _slice)
                all_cnt += 1
                with open(_slice_path, 'r') as f:
                    slice_md5 = get_file_md5(f)
                if slice_md5 not in list(novul_hash_dict.values()):
                    keep_slice_dict[_vul_path].append(_slice_path)
                    keep_cnt += 1
        print('all slices numbers ', all_cnt)
        print('keep slices numbers ', keep_cnt)
        with open(filter_res_path, 'w+') as f:
            json.dump(keep_slice_dict, f) 
    else:
        with open(filter_res_path, 'r') as f:
            keep_slice_dict = json.load(f)
    return keep_slice_dict


def iforest_filter(keep_slice_dict, outliers_fraction):
    vul_keep_slices = []
    slice_name_list = []
    for func_ in keep_slice_dict.keys():
        slice_list = keep_slice_dict[func_]
        for slice_path in slice_list:
            slice_path = slice_path.strip()
            with open(slice_path, 'r') as f:
                slice_emb = json.load(f)
            vul_keep_slices.append(slice_emb[0])
            slice_name_list.append(slice_path)

    rng = np.random.RandomState(42)
    clf = IsolationForest(max_samples=len(vul_keep_slices), random_state=rng, contamination=outliers_fraction)
    clf.fit(vul_keep_slices)

    scores_pred = clf.decision_function(vul_keep_slices)
    threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)  #####

    final_keep_slice_path = []
    for i, _score in enumerate(scores_pred):
        if _score > threshold:
            final_keep_slice_path.append(slice_name_list[i])

    # n_outliers = int(outliers_fraction * len(vul_keep_slices))
    # # xx, yy = np.meshgrid(np.linspace(-7, 7, 50), np.linspace(-7, 7, 50))
    # vul_keep_slices = np.array(vul_keep_slices)
    # Z = clf.decision_function(np.c_[vul_keep_slices.ravel(),vul_keep_slices.ravel()])
    # Z = Z.reshape(vul_keep_slices.shape)
    
    # plt.title("IsolationForest")
    # # plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
    # plt.contourf(vul_keep_slices, vul_keep_slices, Z, levels=np.linspace(Z.min(), threshold, 7), cmap=plt.cm.Blues_r)  #绘制异常点区域，值从最小的到阈值的那部分
    # a = plt.contour(vul_keep_slices, vul_keep_slices, Z, levels=[threshold], linewidths=2, colors='red')  #绘制异常点区域和正常点区域的边界
    # plt.contourf(vul_keep_slices, vul_keep_slices, Z, levels=[threshold, Z.max()], colors='palevioletred')  #绘制正常点区域，值从阈值到最大的那部分
    
    # b = plt.scatter(vul_keep_slices[:-n_outliers, 0], vul_keep_slices[:-n_outliers, 1], c='white',
    #                     s=20, edgecolor='k')
    # c = plt.scatter(vul_keep_slices[-n_outliers:, 0], vul_keep_slices[-n_outliers:, 1], c='black',
    #                     s=20, edgecolor='k')
    # plt.axis('tight')
    # plt.xlim((-7, 7))
    # plt.ylim((-7, 7))
    # plt.legend([a.collections[0], b, c],
    #         ['learned decision function', 'true inliers', 'true outliers'],
    #         loc="upper left")
    # plt.savefig('/home/'+str(outliers_fraction)+'.png', bbox_inches='tight')
    return final_keep_slice_path, clf

def train_kmeans(n, inputs, outliers_fraction,_type_name,_k_fraction):
    kmeans = MiniBatchKMeans(n_clusters=n, random_state=0, verbose=1, max_iter=300, batch_size=6000)
    for i in range(0, len(inputs), 6000):
        kmeans.partial_fit(inputs[i:i+6000])
    if not os.path.exists(os.path.join('/home/output_data_new_cluster_cwe_combine/kmeans_models', '%.3f'%(outliers_fraction))):
        os.makedirs(os.path.join('/home/output_data_new_cluster_cwe_combine/kmeans_models', '%.3f'%(outliers_fraction)))
    model_save_path = os.path.join('/home/output_data_new_cluster_cwe_combine/kmeans_models', '%.3f'%(outliers_fraction), '%.2f'%(_k_fraction)+'_'+_type_name+'.pkl')
    joblib.dump(kmeans, model_save_path)
    return kmeans

_type_name = ["API","pointer","del","array","integer"]
def _slice_type(_slice_path):
    if _slice_path.find('@API') != -1:
        return 0
    elif _slice_path.find('@pointer') != -1:
        return 1
    elif _slice_path.find('@array') != -1:
        return 2
    elif _slice_path.find('@integer') != -1:
        return 3
    elif _slice_path.find('@del') != -1:
        return 4

def cluster_vul_patterns(final_keep_slice_path, outliers_fraction, train_path_txt, test_path_txt, all_txt, iforest, detection_res_path):
    slice_emb_list = [[],[],[],[],[]]
    cluster_counter = [0,0,0,0,0]
    for _slice_path in final_keep_slice_path:
        with open(_slice_path, 'r') as f:
            slice_emb = json.load(f)[0]
            slice_emb_list[_slice_type(_slice_path)].append(slice_emb)
            

    for _type in range(5):
        cluster_counter[_type] = len(slice_emb_list[_type])

    elbow = [[],[],[],[]]

    print('Slice Count: {} {} // API pointer array integer del'.format(len(final_keep_slice_path),cluster_counter))

    for _k_fraction in np.arange(0.05, 0.50, 0.05):
        print('-----> K_means cluster fraction:\t %.2f'%(_k_fraction))
        with open(detection_res_path, 'a+') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['K_means cluster fraction:\t %.2f'%(_k_fraction)])

        kmeans = [None,None,None,None]
        for _type in range(4):
            kmeans[_type] = train_kmeans(int(cluster_counter[_type]*_k_fraction), slice_emb_list[_type], outliers_fraction,_type_name[_type],_k_fraction)
            elbow[_type].append(kmeans[_type].inertia_)
            

        dl_data_dict_path =  os.path.join('/home/output_data_new_cluster_cwe_combine/vul_detector','%.2f_%.2f_data_dict.json'%(_k_fraction,outliers_fraction))
        vul_detection(train_path_txt, test_path_txt, all_txt, kmeans, iforest, dl_data_dict_path, detection_res_path)

        # slice_Y_labels = kmeans.predict(slice_emb_list)
        
        # cluster_counter = np.zeros([_k,5]).astype(int)
        # total_counter = [0,0,0,0,0]

        # for i in range(len(final_keep_slice_path)):
        #     _slice_path = final_keep_slice_path[i]
        #     j = slice_Y_labels[i]
        #     if _slice_path.find('@API') != -1:
        #         cluster_counter[j][0] += 1
        #         total_counter[0] += 1
        #     elif _slice_path.find('@pointer') != -1:
        #         cluster_counter[j][1] += 1
        #         total_counter[1] += 1
        #     elif _slice_path.find('@del') != -1:
        #         cluster_counter[j][2] += 1
        #         total_counter[2] += 1
        #     elif _slice_path.find('@array') != -1:
        #         cluster_counter[j][3] += 1
        #         total_counter[3] += 1
        #     elif _slice_path.find('@integer') != -1:
        #         cluster_counter[j][4] += 1
        #         total_counter[4] += 1
        
        # if not os.path.exists('/home/output_data_new_cluster/kmeans_cluster'):
        #     os.makedirs('/home/output_data_new_cluster/kmeans_cluster')
        # with open(os.path.join('/home/output_data_new_cluster/kmeans_cluster',  str(_k) + '_' + str(outliers_fraction)+ '.csv'),"w") as f:
        #     csv_writer = csv.writer(f)
        #     csv_writer.writerow(total_counter)
        #     csv_writer.writerows(cluster_counter.tolist())

    
    for _type in range(4):
        sns.lineplot(x = np.arange(0.05, 0.50, 0.05).tolist(),y = elbow[_type],color='blue')
        plt.rcParams.update({'figure.figsize':(16,10),'figure.dpi':100})
        plt.title('Elbow Method')
        plt.savefig('/home/output_data_new_cluster_cwe_combine/Elbow_Method'+'%.3f'%(outliers_fraction)+'_'+_type_name[_type]+'.png', bbox_inches='tight')

from tqdm import tqdm

def detector_data_gen(file_list, iforest, cluster_centers, dl_data_dict_path):
    if os.path.exists(dl_data_dict_path):
        with open(dl_data_dict_path, 'r') as f:
            _data_dict = json.load(f)
        return _data_dict

    _data_dict = {}
    err_cnt = 0
    random.shuffle(file_list)
    for _train_func in tqdm(file_list,ascii=True,desc="Train Func"):
        _train_func = _train_func.strip()
        func_dis_tmp = []
        slice_list = os.listdir(_train_func)
        if slice_list == []:
            err_cnt += 1
            continue
        for _slice in slice_list:
            _slice_emb = []
            with open(os.path.join(_train_func, _slice), 'r') as f:
                _slice_emb = json.load(f)
            score = iforest.decision_function(_slice_emb)[0]
            score = (score + 1) / 2 # 规约
            slice_dis = []
            _slice_emb_np = np.array(_slice_emb)
            for _cent in cluster_centers:
                d = np.linalg.norm(_slice_emb_np - _cent)
                slice_dis.append(score/max(d,1e-5))
            func_dis_tmp.append(slice_dis)
        func_emb = list(np.mean(func_dis_tmp, axis=0))
        _data_dict[_train_func] = func_emb
        # if len(_data_dict) == 10: #####
        #     break
    with open(dl_data_dict_path, 'w+') as f:
        json.dump(_data_dict, f)
    print('no slices function:\t', err_cnt)
    return _data_dict

def vul_detection(train_path_txt, test_path_txt, all_txt, kmeans, iforest, dl_data_dict_path, detection_res_path):

    cluster_centers = []
    for _type in range(4):
        cluster_centers.extend(kmeans[_type].cluster_centers_) 

    # TODO:提前划分好
    # with open(train_path_txt, 'r') as f:
    #     train_list = f.readlines()
    # random.shuffle(train_list)
    # with open(test_path_txt, 'r') as f:
    #     test_list = f.readlines()
    # random.shuffle(test_list)

    # train_data_dict = detector_data_gen(train_list, iforest, cluster_centers)
    # test_data_dict = detector_data_gen(test_list, iforest, cluster_centers)

    # TODO:全部数据进行十倍交叉验证
    with open(all_txt, 'r') as f:
        all_file_list = f.readlines()
    
    label_list = []
    embedding_list = []
    train_data_dict = detector_data_gen(all_file_list, iforest, cluster_centers, dl_data_dict_path)
    for _func_name in train_data_dict.keys():
        if '/novul/' in _func_name:
            label_list.append(0)
        else:
            label_list.append(1)
        if train_data_dict[_func_name] == []:
            print(1)
        embedding_list.append(train_data_dict[_func_name])

    detection_res = []
    # print('--------> KNN1:')
    # ml_res = knn_1(embedding_list, label_list)
    # print('F1: ',ml_res[0],'\tPrecision: ', ml_res[1], '\tRecall: ', ml_res[2], '\tAcc: ', ml_res[3])
    # detection_res.append(['knn1', ml_res[0], ml_res[1], ml_res[2], ml_res[3]])
    
    # print('--------> KNN3:')
    # ml_res = knn_3(embedding_list, label_list)
    # print('F1: ',ml_res[0],'\tPrecision: ', ml_res[1], '\tRecall: ', ml_res[2], '\tAcc: ', ml_res[3])
    # detection_res.append(['knn3', ml_res[0], ml_res[1], ml_res[2], ml_res[3]])

    print('--------> randomforest:')
    ml_res = randomforest(embedding_list, label_list)
    print('F1: ',ml_res[0],'\tPrecision: ', ml_res[1], '\tRecall: ', ml_res[2], '\tAcc: ', ml_res[3])
    detection_res.append(['RF', ml_res[0], ml_res[1], ml_res[2], ml_res[3]])

    # print('--------> svm:')
    # ml_res = svm(embedding_list, label_list)
    # print('F1: ',ml_res[0],'\tPrecision: ', ml_res[1], '\tRecall: ', ml_res[2], '\tAcc: ', ml_res[3])
    # detection_res.append(['SVM', ml_res[0], ml_res[1], ml_res[2], ml_res[3]])

    # print('--------> decision_tree:')
    # ml_res = decision_tree(embedding_list, label_list)
    # print('F1: ',ml_res[0],'\tPrecision: ', ml_res[1], '\tRecall: ', ml_res[2], '\tAcc: ', ml_res[3])
    # detection_res.append(['DT', ml_res[0], ml_res[1], ml_res[2], ml_res[3]])

    # print('--------> logistic_regression:')
    # ml_res = logistic_regression(embedding_list, label_list)
    # print('F1: ',ml_res[0],'\tPrecision: ', ml_res[1], '\tRecall: ', ml_res[2], '\tAcc: ', ml_res[3])
    # detection_res.append(['LR', ml_res[0], ml_res[1], ml_res[2], ml_res[3]])
    detection_res.append([])
    print('')
    with open(detection_res_path, 'a+') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(detection_res)
    

from sklearn.neighbors import KNeighborsClassifier
# 利用邻近点方式训练数据
def knn_1(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []
    TPRs = []
    FPRs = []
    TNRs = []
    FNRs = []
    for train_index, test_index in kf.split(X):
        # 将训练集和测试集进行分开
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]
        # ##训练数据###
        clf = KNeighborsClassifier(n_neighbors=1)  # 引入训练方法
        clf.fit(train_X, train_Y)  # 进行填充测试数据进行训练
        y_pred = clf.predict(test_X)  # 预测特征值

        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

        TP = np.sum(np.multiply(test_Y, y_pred))
        FP = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 1)))
        FN = np.sum(np.logical_and(np.equal(test_Y, 1), np.equal(y_pred, 0)))
        TN = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 0)))

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        TNR = TN / (TN + FP)
        FNR = FN / (TP + FN)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)
        TPRs.append(TPR)
        FPRs.append(FPR)
        TNRs.append(TNR)
        FNRs.append(FNR)
    # print(F1s, FPRs)
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys), np.mean(TPRs), np.mean(FPRs), np.mean(TNRs), np.mean(FNRs)]


def knn_3(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)
    i = 0
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []
    TPRs = []
    FPRs = []
    TNRs = []
    FNRs = []
    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(train_X, train_Y)

        #joblib.dump(clf, 'clf_knn_3.pkl')
        y_pred = clf.predict(test_X)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

        #print(f1)
        TP = np.sum(np.multiply(test_Y, y_pred))
        FP = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 1)))
        FN = np.sum(np.logical_and(np.equal(test_Y, 1), np.equal(y_pred, 0)))
        TN = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 0)))

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        TNR = TN / (TN + FP)
        FNR = FN / (TP + FN)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)
        TPRs.append(TPR)
        FPRs.append(FPR)
        TNRs.append(TNR)
        FNRs.append(FNR)
    # print(F1s, FPRs)
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys), np.mean(TPRs), np.mean(FPRs), np.mean(TNRs), np.mean(FNRs)]

from sklearn.ensemble import RandomForestClassifier
def randomforest(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)
    i = 0
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []
    TPRs = []
    FPRs = []
    TNRs = []
    FNRs = []
    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = RandomForestClassifier(max_depth=8, random_state=0)
        clf.fit(train_X, train_Y)

        #joblib.dump(clf, 'clf_randomforest.pkl')
        y_pred = clf.predict(test_X)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

        #print(f1)
        TP = np.sum(np.multiply(test_Y, y_pred))
        FP = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 1)))
        FN = np.sum(np.logical_and(np.equal(test_Y, 1), np.equal(y_pred, 0)))
        TN = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 0)))

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        TNR = TN / (TN + FP)
        FNR = FN / (TP + FN)
        #print(FPR)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)
        TPRs.append(TPR)
        FPRs.append(FPR)
        TNRs.append(TNR)
        FNRs.append(FNR)

    # print(F1s, FPRs)
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys), np.mean(TPRs), np.mean(FPRs), np.mean(TNRs), np.mean(FNRs)]

from sklearn.svm import SVC
def svm(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)
    i = 0
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []
    TPRs = []
    FPRs = []
    TNRs = []
    FNRs = []
    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = SVC(kernel='linear')
        clf.fit(train_X, train_Y)

        #joblib.dump(clf, 'clf_svm.pkl')
        y_pred = clf.predict(test_X)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

        #print(f1)
        TP = np.sum(np.multiply(test_Y, y_pred))
        FP = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 1)))
        FN = np.sum(np.logical_and(np.equal(test_Y, 1), np.equal(y_pred, 0)))
        TN = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 0)))

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        TNR = TN / (TN + FP)
        FNR = FN / (TP + FN)
        #print(FPR)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)
        TPRs.append(TPR)
        FPRs.append(FPR)
        TNRs.append(TNR)
        FNRs.append(FNR)

    # print(F1s, FPRs)
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys), np.mean(TPRs), np.mean(FPRs), np.mean(TNRs), np.mean(FNRs)]

from sklearn import tree
def decision_tree(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)
    i = 0
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []
    TPRs = []
    FPRs = []
    TNRs = []
    FNRs = []
    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = tree.DecisionTreeClassifier()
        clf.fit(train_X, train_Y)

        # 保存model
        #joblib.dump(clf, 'clf_decision_tree.pkl')
        y_pred = clf.predict(test_X)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

        #print(f1)
        TP = np.sum(np.multiply(test_Y, y_pred))
        FP = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 1)))
        FN = np.sum(np.logical_and(np.equal(test_Y, 1), np.equal(y_pred, 0)))
        TN = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 0)))

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        TNR = TN / (TN + FP)
        FNR = FN / (TP + FN)
        #print(FPR)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)
        TPRs.append(TPR)
        FPRs.append(FPR)
        TNRs.append(TNR)
        FNRs.append(FNR)

    # print(F1s, FPRs)
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys), np.mean(TPRs), np.mean(FPRs), np.mean(TNRs), np.mean(FNRs)]

from sklearn.naive_bayes import GaussianNB
def naive_bayes(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)
    i = 0
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []
    TPRs = []
    FPRs = []
    TNRs = []
    FNRs = []
    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = GaussianNB()
        clf.fit(train_X, train_Y)

        #joblib.dump(clf, 'clf_naive_bayes.pkl')
        y_pred = clf.predict(test_X)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

        #print(f1)
        TP = np.sum(np.multiply(test_Y, y_pred))
        FP = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 1)))
        FN = np.sum(np.logical_and(np.equal(test_Y, 1), np.equal(y_pred, 0)))
        TN = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 0)))

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        TNR = TN / (TN + FP)
        FNR = FN / (TP + FN)
        #print(FPR)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)
        TPRs.append(TPR)
        FPRs.append(FPR)
        TNRs.append(TNR)
        FNRs.append(FNR)

    # print(F1s, FPRs)
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys), np.mean(TPRs), np.mean(FPRs), np.mean(TNRs), np.mean(FNRs)]

from sklearn.linear_model import LogisticRegression
def logistic_regression(vectors, labels):
    X = np.array(vectors)
    Y = np.array(labels)

    kf = KFold(n_splits=10)
    i = 0
    F1s = []
    Precisions = []
    Recalls = []
    Accuracys = []
    TPRs = []
    FPRs = []
    TNRs = []
    FNRs = []
    for train_index, test_index in kf.split(X):
        train_X, train_Y = X[train_index], Y[train_index]
        test_X, test_Y = X[test_index], Y[test_index]

        clf = LogisticRegression()
        clf.fit(train_X, train_Y)

        #joblib.dump(clf, 'clf_logistic_regression.pkl')
        y_pred = clf.predict(test_X)
        f1 = f1_score(y_true=test_Y, y_pred=y_pred)
        precision = precision_score(y_true=test_Y, y_pred=y_pred)
        recall = recall_score(y_true=test_Y, y_pred=y_pred)
        accuracy = accuracy_score(y_true=test_Y, y_pred=y_pred)

        #print(f1)
        TP = np.sum(np.multiply(test_Y, y_pred))
        FP = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 1)))
        FN = np.sum(np.logical_and(np.equal(test_Y, 1), np.equal(y_pred, 0)))
        TN = np.sum(np.logical_and(np.equal(test_Y, 0), np.equal(y_pred, 0)))

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        TNR = TN / (TN + FP)
        FNR = FN / (TP + FN)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        Accuracys.append(accuracy)
        TPRs.append(TPR)
        FPRs.append(FPR)
        TNRs.append(TNR)
        FNRs.append(FNR)

    # print(F1s, FPRs)
    return [np.mean(F1s), np.mean(Precisions), np.mean(Recalls), np.mean(Accuracys), np.mean(TPRs), np.mean(FPRs), np.mean(TNRs), np.mean(FNRs)]


def main():
    novul_hash_txt = '/home/input_data/cwe_combine/1_hash_filter/novul_funcs_4hash.txt'
    novul_hash_dict_path = '/home/output_data_new_cluster_cwe_combine/novul_hash_dict.json'
    vul_test_file = '/home/input_data/cwe_combine/2_iforest_filter/vul_funcs_4pattern.txt'
    filter_res_path = '/home/output_data_new_cluster_cwe_combine/filter_res_dict.json'
    train_path_txt = '/home/input_data/cwe_combine/3_detect_models/train.txt'
    test_path_txt = '/home/input_data/cwe_combine/3_detect_models/test.txt'
    all_txt = '/home/input_data/cwe_combine/3_detect_models/all.txt'
    detection_res_path = '/home/output_data_new_cluster_cwe_combine/detection_res.csv'


    novul_hash_dict = novul_hash_dict_get(novul_hash_txt, novul_hash_dict_path)
    keep_slice_dict = hash_filter(novul_hash_dict, vul_test_file, filter_res_path)
    for outliers_fraction in np.arange(0.05, 0.15, 0.01): # 0.05 - 0.5
        print('====> outliers_fraction:\t%.3f'%(outliers_fraction))
        with open(detection_res_path, 'a+') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['outliers_fraction:\t%.3f'%(outliers_fraction)])
        final_keep_slice_path, iforest = iforest_filter(keep_slice_dict, outliers_fraction)
        cluster_vul_patterns(final_keep_slice_path, outliers_fraction, train_path_txt, test_path_txt, all_txt, iforest, detection_res_path)

if __name__ == '__main__':
    main()