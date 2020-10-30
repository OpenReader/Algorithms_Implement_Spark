from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from itertools import combinations
import math
import random
import time
import sys


def main(arg):

    conf = SparkConf()
    conf.set("spark.app.name", "haochen_guo_competition")
    conf.set("spark.master", "local[*]")
    sc = SparkContext(conf=conf)


    # In[4]:


    train_file_name = arg[0]
    test_file_name = arg[1]
    output_file_name = arg[2]


    # In[ ]:


    # input
    input_rdd = sc.textFile(train_file_name+"yelp_train.csv")
    header = input_rdd.first()
    u_b_s = input_rdd.filter(lambda s: s != header).map(lambda s: s.strip('\n').split(','))

    u_list = u_b_s.map(lambda s: s[0]).distinct().collect()
    u_id = dict()
    for i, v in enumerate(u_list):
        u_id[v] = i
    b_list = u_b_s.map(lambda s: s[1]).distinct().collect()
    b_id = dict()
    for i, v in enumerate(b_list):
        b_id[v] = i

    uid = sc.broadcast(u_id)
    bid = sc.broadcast(b_id)
        
    u_b_r = u_b_s.map(lambda s: (u_id[s[0]], b_id[s[1]], float(s[2])))


    # In[ ]:


    # avg of all
    globe_avg = u_b_r.map(lambda s: s[2]).mean()
    g_a = sc.broadcast(globe_avg)


    # In[ ]:


    # load avg of each busi
    import json
    b_rdd = sc.textFile(arg[0]+"business.json").map(lambda s: (json.loads(s)["business_id"],json.loads(s)["stars"]))
    b_avg_f = b_rdd.filter(lambda s: s[0] in bid.value).map(lambda s: (-1, bid.value[s[0]], float(s[1])))


    # In[ ]:


    # load avg of each user
    u_rdd = sc.textFile(arg[0]+"user.json").map(lambda s: (json.loads(s)["user_id"], json.loads(s)["average_stars"]))
    u_avg_f = u_rdd.filter(lambda s: s[0] in uid.value).map(lambda s: (uid.value[s[0]], -1, float(s[1])))


    # In[ ]:


    # union b_a_f
    u_b_r = u_b_r.union(b_avg_f)
    # union u_a_f
    u_b_r = u_b_r.union(u_avg_f)


    # In[ ]:


    # avg of each user
    u_avg = u_b_r.map(lambda s: (s[0], (s[2], 1))).reduceByKey(lambda k1, k2: (k1[0]+k2[0], k1[1]+k2[1])).map(lambda s: (s[0], s[1][0]/s[1][1]))
    u_a_d = dict()
    for t in u_avg.collect():
        u_a_d[t[0]] = t[1]
    ua = sc.broadcast(u_a_d)


    # In[ ]:


    # avg of each busi
    b_avg = u_b_r.map(lambda s: (s[1], (s[2], 1))).reduceByKey(lambda k1, k2: (k1[0]+k2[0], k1[1]+k2[1])).map(lambda s: (s[0], s[1][0]/s[1][1]))
    b_a_d = dict()
    for t in b_avg.collect():
        b_a_d[t[0]] = t[1]
    ba = sc.broadcast(b_a_d)


    # In[ ]:


    # union b_a
    b_avg_train = b_avg.map(lambda s: (-2, s[0], s[1]))
    u_b_r = u_b_r.union(b_avg_train)
    # union u_a
    u_avg_train = u_avg.map(lambda s: (s[0], -2, s[1]))
    u_b_r = u_b_r.union(u_avg_train)


    # In[ ]:


    # Add rest
    baf_l = b_avg_f.map(lambda s: s[2]).collect()
    uaf_l = u_avg_f.map(lambda s: s[2]).collect()
    ba_l = b_avg_train.map(lambda s: s[2]).collect()
    ua_l = u_avg_train.map(lambda s: s[2]).collect()
    a11 = (sum(baf_l)+sum(uaf_l)) / (len(baf_l)+len(uaf_l))
    a12 = (sum(ba_l)+sum(uaf_l)) / (len(ba_l)+len(uaf_l))
    a21 = (sum(baf_l)+sum(ua_l)) / (len(baf_l)+len(ua_l))
    a22 = (sum(ba_l)+sum(ua_l)) / (len(ba_l)+len(ua_l))

    rest = [(-1,-1,a11), (-1,-2,a12), (-2,-1,a21), (-2,-2,a22)]
    u_b_r = u_b_r.union(sc.parallelize(rest))


    # In[ ]:
    # test data
    def classify(line):
        arr = line.strip('\n').split(',')
        if arr[0] in uid.value and arr[1] in bid.value:
            return (0, uid.value[arr[0]], bid.value[arr[1]], float(arr[2]))
        if arr[0] in uid.value:
            return (1, arr[0], arr[1], float(arr[2]))
        if arr[1] in bid.value:
            return (2, arr[0], arr[1], float(arr[2]))
        return (3, arr[0], arr[1], float(arr[2]))

    def outPredict(s):
        if s[0] == 1:
            return (s[1], s[2], ua.value[uid.value[s[1]]])
        if s[0] == 2:
            return (s[1], s[2], ba.value[bid.value[s[2]]])
        return (s[1], s[2], g_a.value)

    test_rdd = sc.textFile(test_file_name).filter(lambda s: s != header).map(classify)
    in_train = test_rdd.filter(lambda s: s[0] == 0).map(lambda s: (s[1], s[2], s[3]))
    out_train_res = test_rdd.filter(lambda s: s[0] != 0).map(outPredict).collect()

    # Item-based

    def IBPCC(b1, b2):
        d1 = bd.value[b1]
        d2 = bd.value[b2]
        w, r1, r2, n, s1, s2 = 0, 0, 0, 0, 0, 0
        for k in d1:
            if k in d2:
                n += 1
                r1 += d1[k]
                r2 += d2[k]
        if n == 0:
            return 0
    #     r1 = r1 / n
    #     r2 = r2 / n
    #     r1 = ba.value[b1]
    #     r2 = ba.value[b2]
        r1 = r2 = g_a.value
        for k in d1:
            if k in d2:
                w += (d1[k]-r1) * (d2[k]-r2)
                s1 += (d1[k] - r1) ** 2
                s2 += (d2[k] - r2) ** 2
        if w == 0:
            return 0
        return w / (math.sqrt(s1) * math.sqrt(s2))

    # co_data: (uid, ([(train_bid, rating)], test_bid))
    def IBpredict(co_data):
        test_bid = co_data[1][1]
        sum_r_w = 0
        sum_aw = 0
        for t in co_data[1][0]:
            w = IBPCC(test_bid, t[0])
            if w > 0:
                sum_r_w += t[1] * w
                sum_aw += w
        if sum_r_w == 0:
            return co_data[0], test_bid, ba.value[test_bid]
        rating = sum_r_w / sum_aw
    #     if rating > 5.0:
    #         rating = 5.0
    #     if rating < 1.0:
    #         rating = 1.0
        return co_data[0], test_bid, rating

    b_train = u_b_r.map(lambda s: (s[1], [(s[0], s[2])])).reduceByKey(lambda a, b: a+b).collect()
    b_d = dict()
    for t in b_train:
        d = dict()
        for e in t[1]:
            d[e[0]] = e[1]
        b_d[t[0]] = d
    bd = sc.broadcast(b_d)

    u_train = u_b_r.map(lambda s: (s[0], [(s[1], s[2])])).reduceByKey(lambda a, b: a+b)
    u_test = in_train.map(lambda s: (s[0], s[1]))
    t_t = u_train.join(u_test)
    res = t_t.map(IBpredict).filter(lambda s: s[0] >= 0 and s[1] >= 0).map(lambda s: (u_list[s[0]], b_list[s[1]], s[2])).collect()


    # In[ ]:


    # save
    with open(output_file_name, 'w') as o_f:
        o_f.write("user_id, business_id, prediction\n")
        for triple in res:
            o_f.write("%s,%s,%f\n" % (triple[0], triple[1], triple[2]))
        for triple in out_train_res:
            o_f.write("%s,%s,%f\n" % (triple[0], triple[1], triple[2]))


    # In[ ]:


    # RMSE
    # predictions
    p_rdd = sc.textFile(output_file_name)
    p_header = p_rdd.first()
    predictions = p_rdd.filter(lambda s: s != p_header).map(lambda s: s.strip('\n').split(','))    .map(lambda s: ((s[0], s[1]), float(s[2])))
    # validation
    v_rdd = sc.textFile(test_file_name)
    v_header = v_rdd.first()
    ratings = v_rdd.filter(lambda s: s != v_header).map(lambda s: s.strip('\n').split(','))    .map(lambda s: ((s[0], s[1]), float(s[2])))

    ratesAndPreds = ratings.join(predictions)
    RMSE = math.sqrt(ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    print("******************************")
    print("Root Mean Squared Error = " + str(RMSE))
    print("******************************")

if __name__ == "__main__":
    main(sys.argv[1:])