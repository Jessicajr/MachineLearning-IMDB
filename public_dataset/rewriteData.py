import csv
f = open('test_data.csv','r',encoding='utf-8')
fw = open('new_test_data.csv','w',encoding='utf-8',newline="")
fw_csv = csv.writer(fw)
for i,item in enumerate(csv.reader(f)):
    if i==0:
        fw_csv.writerow([j for j in item])
    else:
        if item[2]=='positive':
            fw_csv.writerow([item[0],item[1],1])
        else:
            fw_csv.writerow([item[0], item[1], 0])
f.close()
fw.close()
# f1 = open('s_train.csv','r',encoding='utf-8')
# f2 = open('s_test.csv','r',encoding='utf-8')
# fw1 = open('train_s.csv','w',newline="",encoding='utf-8')
# fw2 = open('test_s.csv','w',newline="",encoding='utf-8')
# fw1_csv = csv.writer(fw1)
# fw2_csv = csv.writer(fw2)
# for i,item in enumerate(csv.reader(f1)):
#     if i==0:
#         fw1_csv.writerow([j for j in item])
#     else:
#         if item[2]=='positive':
#             fw1_csv.writerow([item[0],item[1],1])
#         else:
#             fw1_csv.writerow([item[0], item[1], 0])
# for i,item in enumerate(csv.reader(f2)):
#     if i==0:
#         fw2_csv.writerow([j for j in item])
#     else:
#         if item[2]=='positive':
#             fw2_csv.writerow([item[0],item[1],1])
#         else:
#             fw2_csv.writerow([item[0], item[1], 0])
# f2.close()
# f1.close()
# fw1.close()
# fw2.close()

