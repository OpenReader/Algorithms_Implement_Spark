from pyspark import SparkConf
from pyspark import SparkContext
from itertools import combinations
import random
import sys




def main(arg):

	conf = SparkConf()
	conf.set("spark.app.name", "haochen_guo_task1")
	conf.set("spark.master", "local[*]")
	sc = SparkContext(conf=conf)


	# In[4]:


	# input_file_name = "/mnt/hgfs/Documents/INF_553/assignments3/yelp_train.csv"
	# output_file_name = "/mnt/hgfs/Documents/INF_553/assignments3/task1_output.csv"
	input_file_name = arg[0]
	output_file_name = arg[1]


	# In[5]:


	input_rdd = sc.textFile(input_file_name)
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
	u_len = len(u_list)
	b_len = len(b_list)


	# In[6]:


	u_m = u_b_s.map(lambda s: (b_id[s[1]], [u_id[s[0]]])).reduceByKey(lambda a, b: a+b)


	# In[7]:


	um_d = dict()
	for busi in u_m.collect():
	    um_d[busi[0]] = busi[1]
	um = sc.broadcast(um_d)


	# In[8]:


	# hash table
	m = 99
	b = 33
	r = int(m/b)

	raw_list = [i for i in range(u_len)]
	h_t_main = set()
	while True:
	    random.shuffle(raw_list)
	    h_t_main.add(tuple(raw_list))
	    if len(h_t_main) == m:
	        break
	h_t_main = list(h_t_main)


	# In[9]:


	h_t = sc.broadcast(h_t_main)


	# In[10]:


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


	# In[11]:


	def jaccard(candi_list):
	    pairs = combinations(candi_list[1], 2)
	    for p in pairs:
	        a = set(um.value[p[0]])
	        b = set(um.value[p[1]])
	        i = a & b
	        u = a | b
	        yield (p[0], p[1]), float(len(i)/len(u))


	# In[12]:


	band_dic = u_m.flatMap(minihash).reduceByKey(lambda a, b: a+b).filter(lambda s: len(s[1])>1)


	# In[13]:


	sim_pair = band_dic.flatMap(jaccard).filter(lambda s: s[1] >= 0.5).distinct().collect()


	# In[14]:


	result = sorted([[(min(b_list[t[0][0]], b_list[t[0][1]]), max(b_list[t[0][0]], b_list[t[0][1]])), t[1]] for t in sim_pair])


	# valication
	
	# res = [(t[0][0], t[0][1]) for t in result]
	# length = len(res)
	# ans = sc.textFile("/mnt/hgfs/Documents/INF_553/assignments3/pure_jaccard_similarity.csv")
	# header = ans.first()
	# ans = ans.filter(lambda x: x != header).map(lambda x: x.split(',')).map(lambda x : (x[0],x[1])).collect()
	# right_length = len(ans)

	# common = 0
	# candidates_set = set(res)
	# for pair in ans:
	#     if pair in candidates_set:
	#         common += 1
	# precision = float(common)/length
	# recall = float(common)/right_length
	# print("##################################")
	# print(len(res)," predict length")
	# print(right_length,"ans_length")
	# print(common," common length")
	# print("precision = "+str(precision))
	# print("recall = "+str(recall))
	# print("##################################")

	# In[15]:


	with open(output_file_name, 'w') as o_f:
	    o_f.write("business_id_1, business_id_2, similarity\n")
	    for pair in result:
	        o_f.write("%s,%s,%f\n" % (pair[0][0], pair[0][1], pair[1]))


if __name__ == "__main__":
    main(sys.argv[1:])