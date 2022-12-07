
import numpy as np
import time
import faiss 

def index(Data, index,Path):
    np.random.seed(1234)             # make reproducible
    # datapath = Data_path            
    xb = Data
    d = xb.shape[1]                         # dimension
    nb = xb.shape[0]                        # database size, 1M

    xb = xb.astype(np.float32)
    print (type(xb[0,0]))
    res = faiss.StandardGpuResources()  # use a single GPU

    ## Using an IVF index
    t1 = time.time()

    index_ivf = faiss.index_factory(d, index, faiss.METRIC_INNER_PRODUCT) #'IVF16384,PQ96'
    print (res)
    gpu_index_ivf = faiss.index_cpu_to_gpu(res, 0, index_ivf) 

    assert not index_ivf.is_trained
    print ("traning")
    index_ivf.train(xb)        # train with nb vectors
    assert index_ivf.is_trained

    print ("adding")
    index_ivf.add(xb)          # add vectors to the index
    print("ntotal after ivf: ",index_ivf.ntotal)

    print ("total train time: ", time.time()-t1)

    saveLoc=Path;

    print ("saving at ", saveLoc)
#     faiss.write_index(index_ivf, saveLoc)
    faiss.write_index(faiss.index_gpu_to_cpu(index_ivf), saveLoc)
    t2 = time.time()
    print ("total code time: ", t2-t1)