import numpy as np

def read_file(file_name, index = 1):
    lines = open(file_name, "r").readlines()
    read_list = {}
    for l in lines:
        tmps = l.strip() 
        read_list[int(tmps.split("\t")[index])] = []
    for l in lines:
        tmps = l.strip()
        read_list[int(tmps.split("\t")[index])] += eval(tmps.split("\t")[-1])
    return read_list

def evaluate_user_perspective(user_perspective_data, u_i_expl_dict):
    pres = []
    recs = []
    f1s = []

    for u_i, gt_features in user_perspective_data.items():
        if u_i in u_i_expl_dict:
            
            TP = 0
            pre_features = u_i_expl_dict[u_i]

            if pre_features and gt_features:
                for feature in pre_features:
                    if feature in gt_features:
                        TP += 1

                pre = TP / len(pre_features)
                rec = TP / len(gt_features)
                if (pre + rec) != 0:
                    f1 = (2 * pre * rec) / (pre + rec)
                else:
                    f1 = 0
            pres.append(pre)
            recs.append(rec)
            f1s.append(f1)
    ave_pre = np.mean(pres)
    ave_rec = np.mean(recs)
    ave_f1 = np.mean(f1s)
    return ave_pre, ave_rec, ave_f1

baseline = './GT_attributes/GT_counter_attributes_yelp2018.txt'
predict = './explanation/attributes_yelp2018.txt' 
baseline_li = read_file(baseline, index=0) 
predict_li = read_file(predict, index=0)

print('Evaluation files:', baseline, predict)

print (evaluate_user_perspective(baseline_li, predict_li))