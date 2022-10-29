import numpy as np
import time
import faiss

def query(dim,data_size,nqueries,GPU,test_path,nns_path, Index_path, K):
#change these params
    d = dim                                      # dimension
    nb = data_size                               # database size, 1B
    nq = nqueries                                # nb of queries
    np.random.seed(1234)                         # make reproducible
    gpu = False

    test_datapath = test_path    
    nns_datapath =  nns_path     
    test = np.load(test_datapath).astype(np.float32)
    nns = np.load(nns_datapath)
    # print (type(test[0,0]))

    index_path = Index_path 

    cpu_index = faiss.read_index(index_path)
    index_ivf = faiss.extract_index_ivf(cpu_index)
    if gpu:
        co = faiss.GpuMultipleClonerOptions()
        gpu_index_ivf = faiss.index_cpu_to_all_gpus(index_ivf, co)
    else:
        gpu_index_ivf = index_ivf

    k = K                                                          # we want to see 4 nearest neighbors
    f1 = open(index_path.split('.')[0] + 'sift-1b-Faiss.txt', "a")

    # Nprobe =  list(range(1,10,2))+ list(range(10,100,20))
    Nprobe=[1]
    for nprobe in Nprobe:
        if gpu:
            for i in range(gpu_index_ivf.count()):
                faiss.downcast_index(gpu_index_ivf.at(i)).nprobe = nprobe
        else:
            gpu_index_ivf.nprobe = nprobe

        labels = np.empty([nq, k])
        D = np.empty([nq, k])
        t1 = time.time()

        # batch
        for j in range(nq):
            D[j:j+1,], labels[j:j+1,] = gpu_index_ivf.search(test[j:(j+1),:], k)  # actual search

        t2 = time.time()
        qt = t2-t1
        # print ("total test time: ", t2-t1)
        return D,labels;

    f1.close()