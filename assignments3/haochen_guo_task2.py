from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from itertools import combinations
import math
import random
import sys
# import time


# In[3]:

def main(arg):
    # start = time.time()

    conf = SparkConf()
    conf.set("spark.app.name", "haochen_guo_task2")
    conf.set("spark.master", "local[*]")
    sc = SparkContext(conf=conf)


    # In[4]:

    train_file_name = arg[0]
    test_file_name = arg[1]
    case_id = int(arg[2])
    output_file_name = arg[3]

    # train_file_name = "/mnt/hgfs/Documents/INF_553/assignments3/yelp_train.csv"
    # test_file_name = "/mnt/hgfs/Documents/INF_553/assignments3/yelp_val.csv"
    # output_file_name = "/mnt/hgfs/Documents/INF_553/assignments3/task2_case1_output.csv"


    # In[5]:


    # input
    input_rdd = sc.textFile(train_file_name)
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


    # In[6]:


    # avg of each user
    u_avg = u_b_r.map(lambda s: (s[0], (s[2], 1))).reduceByKey(lambda k1, k2: (k1[0]+k2[0], k1[1]+k2[1]))    .map(lambda s: (s[0], s[1][0]/s[1][1])).collect()
    u_a_d = dict()
    for t in u_avg:
        u_a_d[t[0]] = t[1]
    ua = sc.broadcast(u_a_d)


    # In[7]:


    # avg of each busi
    b_avg = u_b_r.map(lambda s: (s[1], (s[2], 1))).reduceByKey(lambda k1, k2: (k1[0]+k2[0], k1[1]+k2[1]))    .map(lambda s: (s[0], s[1][0]/s[1][1])).collect()
    b_a_d = dict()
    for t in b_avg:
        b_a_d[t[0]] = t[1]
    ba = sc.broadcast(b_a_d)


    # In[8]:


    globe_avg = 0
    for k in b_a_d:
        globe_avg += b_a_d[k]
    globe_avg = globe_avg/len(b_a_d)
    g_a = sc.broadcast(globe_avg)


    # In[9]:


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

    test_rdd = sc.textFile(test_file_name).filter(lambda s: s != header)    .map(classify)
    in_train = test_rdd.filter(lambda s: s[0] == 0).map(lambda s: (s[1], s[2], s[3]))
    out_train_res = test_rdd.filter(lambda s: s[0] != 0).map(outPredict).collect()


    # In[22]:


    # Model
    if case_id == 1:
        m_m = u_b_r.map(lambda s: Rating(s[0], s[1], float(s[2])))

        rank = 3
        numIterations = 6
        model = ALS.train(m_m, rank, numIterations)

        testdata = in_train.map(lambda s: (s[0], s[1]))
        res = model.predictAll(testdata).map(lambda s: (u_list[s[0]], b_list[s[1]], s[2])).collect()



    # User-based
    if case_id == 2:
        def UBPCC(u1, u2):
            d1 = ud.value[u1]
            d2 = ud.value[u2]
            w, r1, r2, n, s1, s2 = 0, 0, 0, 0, 0, 0
            for k in d1:
                if k in d2:
                    n += 1
                    r1 += d1[k]
                    r2 += d2[k]
            if n == 0:
                return 0
            r1 = r1 / n
            r2 = r2 / n
            for k in d1:
                if k in d2:
                    w += (d1[k]-r1) * (d2[k]-r2)
                    s1 += (d1[k] - r1) ** 2
                    s2 += (d2[k] - r2) ** 2
            if w == 0:
                return 0
            return w / (math.sqrt(s1) * math.sqrt(s2))
                

        # co_data: (bid, ([(train_uid, rating)], test_uid))
        def UBpredict(co_data):
            test_uid = co_data[1][1]
            sum_r_w = 0
            sum_aw = 0
            for t in co_data[1][0]:
                w = UBPCC(test_uid, t[0])
                sum_r_w += (t[1]-ua.value[t[0]]) * w
                sum_aw += math.fabs(w)
            if sum_r_w == 0:
                return test_uid, co_data[0], ua.value[test_uid]
            rating = ua.value[test_uid] + (sum_r_w / sum_aw)
            if rating > 5.0:
                rating = 5.0
            elif rating < 1.0:
                rating = 1.0
            return test_uid, co_data[0], rating

        u_train = u_b_r.map(lambda s: (s[0], [(s[1], s[2])])).reduceByKey(lambda a, b: a+b).collect()
        u_d = dict()
        for t in u_train:
            d = dict()
            for e in t[1]:
                d[e[0]] = e[1]
            u_d[t[0]] = d
        ud = sc.broadcast(u_d)

        b_train = u_b_r.map(lambda s: (s[1], [(s[0], s[2])])).reduceByKey(lambda a, b: a+b)
        b_test = in_train.map(lambda s: (s[1], s[0]))
        t_t = b_train.join(b_test)
        res = t_t.map(UBpredict).map(lambda s: (u_list[s[0]], b_list[s[1]], s[2])).collect()


    # Item-based
    if case_id == 3:
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
            r1 = r1 / n
            r2 = r2 / n
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
                if w > 1:
                    print("**************")
                    print(w)
                    print("**************")
                    sum_r_w += t[1] * w
                    sum_aw += w
            if sum_r_w == 0:
                return co_data[0], test_bid, ba.value[test_bid]
            rating = sum_r_w / sum_aw
        #     if rating > 5.0:
        #         rating = 5.0
            if rating < 1.0:
                rating = 1.0
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
        res = t_t.map(IBpredict).map(lambda s: (u_list[s[0]], b_list[s[1]], s[2])).collect()



    # Item-based with LSH
    if case_id == 4:
        # hash table
        m = 99
        b = 33
        r = int(m/b)

        u_len = len(u_list)
        b_len = len(b_list)

        raw_list = [i for i in range(u_len)]
        h_t_main = set()
        while True:
            random.shuffle(raw_list)
            h_t_main.add(tuple(raw_list))
            if len(h_t_main) == m:
                break
        h_t_main = list(h_t_main)
        h_t = sc.broadcast(h_t_main)

        # minihash
        def minihash(b_vec):
            m = 99
            b = 33
            r = int(m/b)
            sig = [len(h_t.value[0]) for i in range(m)]
            for u_l in b_vec[1]:
                sig = [min(v, h_t.value[i][u_l]) for i, v in enumerate(sig)]
            for i in range(b):
                band = [sig[i*r + j] for j in range(r)]
                band.append(i)
                yield tuple(band), [b_vec[0]]
                
        def jaccard(candi_list):
            pairs = combinations(candi_list[1], 2)
            for p in pairs:
                a = set(um.value[p[0]])
                b = set(um.value[p[1]])
                i = a & b
                u = a | b
                yield (p[0], p[1]), float(len(i)/len(u))

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
            r1 = r1 / n
            r2 = r2 / n
            for k in d1:
                if k in d2:
                    w += (d1[k]-r1) * (d2[k]-r2)
                    s1 += (d1[k] - r1) ** 2
                    s2 += (d2[k] - r2) ** 2
            if w == 0:
                return 0
            return math.fabs(w) / (math.sqrt(s1) * math.sqrt(s2))        

        # co_data: (uid, ([(train_bid, rating)], test_bid))
        def IBLSHpredict(co_data):
            test_bid = co_data[1][1]
            sum_r_w = 0
            sum_aw = 0
            for t in co_data[1][0]:
                if (min(test_bid, t[0]), max(test_bid, t[0])) in ss.value:
                    w = IBPCC(test_bid, t[0])
                    sum_r_w += t[1] * w
                    sum_aw += math.fabs(w)
            if sum_r_w == 0:
                return co_data[0], test_bid, ba.value[test_bid]
            rating = sum_r_w / sum_aw
            if rating < 1.0:
                rating = 1.0
            return co_data[0], test_bid, rating

        u_m = u_b_r.map(lambda s: (s[1], [s[0]])).reduceByKey(lambda a, b: a+b)

        um_d = dict()
        for busi in u_m.collect():
            um_d[busi[0]] = busi[1]
        um = sc.broadcast(um_d)

        band_dic = u_m.flatMap(minihash).reduceByKey(lambda a, b: a+b).filter(lambda s: len(s[1])>1)
        sim_pair = band_dic.flatMap(jaccard).filter(lambda s: s[1] >= 0.5).distinct().collect()

        sim_set = set([(min(t[0][0], t[0][1]), max(t[0][0], t[0][1])) for t in sim_pair])
        ss = sc.broadcast(sim_set)

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
        res = t_t.map(IBLSHpredict).map(lambda s: (u_list[s[0]], b_list[s[1]], s[2])).collect()




    # save
    with open(output_file_name, 'w') as o_f:
        o_f.write("user_id, business_id, prediction\n")
        for triple in res:
            o_f.write("%s,%s,%f\n" % (triple[0], triple[1], triple[2]))
        for triple in out_train_res:
            o_f.write("%s,%s,%f\n" % (triple[0], triple[1], triple[2]))

    # print("Run Time = " + str(time.time()-start))
    
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
    print("***************************")
    print("Root Mean Squared Error = " + str(RMSE))
    print("***************************")

if __name__ == "__main__":
    main(sys.argv[1:])



