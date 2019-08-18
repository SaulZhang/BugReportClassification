'''
2019-7-29 13:00 by @saulzhang
处理文本得到token，训练Word2vec模型
'''
#encoding:utf-8
import csv
import logging  #日记相关配置模块

from gensim.models import word2vec
from nltk.stem.porter import PorterStemmer


#读取csv文件
csv_file_0 = csv.reader(open('../dataset/input_sec/openstack_1.csv'))
csv_file_1 = csv.reader(open('../dataset/input_sec/openstack_0.csv'))

segment = "../data_openstack/segemnt2word_train_and_testset_pre.txt"#存放切割完的Token的文件

file_write = open(segment,"w")

word_list = []
line_cnt = 0

porter_stemmer = PorterStemmer()

for idx,(data,label) in enumerate(csv_file_0):
	# if idx > 10:break
	if idx % 1000 == 0:
		print(idx)
	data = data.lower()
	# print(data)
	# print()
	data = data.translate(str.maketrans('1234567890', '0000000000',"★！？。｡＂＃＄％＆＇（）＊＋，－：；＜＝＞＠＼＾｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏[!\"#$%&\'()*+,;<=>?@\\^`{|}~+" ))
	li = data.split(" ")
	li_1 = []
	for i in li:
		if("http" in i or "https" in i):
			li_1.append("websiteaddress")
		elif(('0' in i and ('-' in i)) or ('0' in i and (':' in i))):
			li_1.append("numbersequence")
		elif('/' in i):
			li_1.append("directory")
		elif(len(i)>=25):
			li_1.append("longsequencenumber")
		elif('[' in i and ']' in i):
			li_1.append("parameter_refer")
		elif(i.isdigit()):
			li_1.append("digit")
		elif(i.count('.')==3 and i.count('0')>=4):
			li_1.append("IPV4")
		elif(len(i.split('.'))>=2):
			li_1 += i.split('.')
		else:
			li_1.append(i)#porter_stemmer.stem(i)
	
	data = " ".join(str(i) for i in li_1)
	# print(data)
	# print()
	# print()
	# print()
	# print()
	file_write.write(data+"\n")
	line_cnt+=1

for idx,(data,label) in enumerate(csv_file_1):
	# if idx > 10:break
	data = data.lower()
	# print(data)
	# print()
	data = data.translate(str.maketrans('1234567890', '0000000000',"★！？。｡＂＃＄％＆＇（）＊＋，－：；＜＝＞＠＼＾｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏[!\"#$%&\'()*+,;<=>?@\\^`{|}~+" ))
	li = data.split(" ")
	li_1 = []
	for i in li:
		if("http" in i or "https" in i):
			li_1.append("websiteaddress")
		elif(('0' in i and ('-' in i)) or ('0' in i and (':' in i))):
			li_1.append("numbersequence")
		elif('/' in i):
			li_1.append("directory")
		elif(len(i)>=25):
			li_1.append("longsequencenumber")
		elif('[' in i and ']' in i):
			li_1.append("parameter_refer")
		elif(i.isdigit()):
			li_1.append("digit")
		elif(i.count('.')==3 and i.count('0')>=4):
			li_1.append("IPV4")
		elif(len(i.split('.'))>=2):
			li_1 += i.split('.')
		else:
			li_1.append(i)#porter_stemmer.stem(i)
	
	data = " ".join(str(i) for i in li_1)
	# print(data)
	# print()
	# print()
	# print()
	# print()
	file_write.write(data+"\n")
	line_cnt+=1

print("The size of token list is:",line_cnt)


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,filename='pre-train.log')
sentences = word2vec.LineSentence("../data_openstack/segemnt2word_train_and_testset_pre.txt")
# model = word2vec.Word2Vec(sentences, size=150)
print("Begin train!")
model = word2vec.Word2Vec(sentences, sg=1, size=150,  window=5,  min_count=3,  negative=3, sample=0.001, hs=1, workers=4)

#保存模型，供日後使用
model.save("../word2vec_model/openstack/word2vec_pre.model")
#模型讀取方式
