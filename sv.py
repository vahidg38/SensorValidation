#by Vahid
#https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_autoencoder/
import numpy as np

from models import *

from arguments import *
from sklearn.decomposition import PCA
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, TableColumn, DataTable
from bokeh.io import show
#from ConvolutionalModels import *
TO_TRAIN=False
TO_TEST=True


args = get_args()


np.random.seed(4)
path=args.path



"""# **Data loading**"""
train_data, test_data = data_loading(mat_file=path, train_test_ratio=4)
# train_data, test_data = data_loading_csv()

mu, sigma = 0, 0.1
noise = np.random.normal(mu, sigma, len(train_data)*5).reshape(-1,5)
# noise = np.random.normal(mu, sigma, len(train_data)*4).reshape(-1,4)
# noise=np.insert(noise, 0, [0], axis=1)
train_noisy= train_data + 100*noise



# Generating  train and real_test , and noisy train dataframes
sensors_=['PM25', 'PM10', 'CO2', 'Temp', 'Humidity']
train_ori, test_ori, train_n_ori = create_dataframe(train_data, sensors=sensors_), create_dataframe(test_data,sensors=sensors_), create_dataframe(train_noisy,sensors=sensors_)
print(train_ori.describe())
print(test_ori.describe())
print(train_n_ori.describe())

#normalizing the data

train, test, train_n = normalize(train_ori, test_ori, train_n_ori)
# print(train.describe())
# print(test.describe())
# print(train_n.describe())

# Generating  faulty_test dataframe
test_faulty= fault_generation(test.copy(), type=args.failure, sensor=args.fsensor, magnitude=args.fmagnitude, start=args.fstart, stop=args.fstop)

#test_faulty=None

#
MSE_=[]  # for performance measurement table data
RR_=[]
MAE_=[]
MAPE_=[]

args.model="integrated"
integrated_model= model(args, train, train_n, test, test_faulty)

if  TO_TRAIN:
    integrated_model.train_model()

if    TO_TEST:

   # integrated_model.reconstruct(train,train_ori, description="train")
   # integrated_model.reconstruct(test,test_ori ,description="test")

    z, x,e = integrated_model.reconstruct(test_faulty, test_ori, description=args.failure)
    print(f"MSE for {args.model}:  {MSE(z, x)}")
    print(f"RR for {args.model}:  {RR(z, x)}")
    print(f"MAE for {args.model}:  {MAE(z, x)}")
    print(f"MAPE for {args.model}:  {MAPE(z, x)}")
    print(e)
    m=e>find_threshold_method_two(e)
    print(f"Threshold:{m.sum()}")
    print(type(m))
    m.to_csv("error.csv")
    MSE_.append(MSE(z, x))
    RR_.append(RR(z, x))
    MAE_.append(MAE(z, x))
    MAPE_.append(MAPE(z, x))
raise Exception("Sorry, no numbers below zero")

args.model="MAE"
MeAE= model(args, train, train_n, test, test_faulty)
#MAE.optimization()
if   TO_TRAIN:
    MeAE.train_model()
if   TO_TEST:
    MeAE.reconstruct(train,train_ori, description="train")
    MeAE.reconstruct(test,test_ori ,description="test")

    z, x=MeAE.reconstruct(test_faulty,test_ori ,description=args.failure)
    print(f"MSE for {args.model}:  {MSE(z,x)}")
    print(f"RR for {args.model}:  {RR(z,x)}")
    print(f"MAE for {args.model}:  {MAE(z,x)}")
    print(f"MAPE for {args.model}:  {MAPE(z,x)}")

    MSE_.append(MSE(z, x))
    RR_.append(RR(z, x))
    MAE_.append(MAE(z, x))
    MAPE_.append(MAPE(z, x))



args.model="AE"
AE= model(args, train, train_n, test, test_faulty)
#AE.optimization()
if   TO_TRAIN:
    AE.train_model()
if   TO_TEST:
    AE.reconstruct(train,train_ori, description="train")
    AE.reconstruct(test, test_ori, description="test")

    z,x=AE.reconstruct(test_faulty,test_ori ,description=args.failure)
    print(f"MSE for {args.model}:  {MSE(z, x)}")
    print(f"RR for {args.model}:  {RR(z, x)}")
    print(f"MAE for {args.model}:  {MAE(z, x)}")
    print(f"MAPE for {args.model}:  {MAPE(z, x)}")

    MSE_.append(MSE(z, x))
    RR_.append(RR(z, x))
    MAE_.append(MAE(z, x))
    MAPE_.append(MAPE(z, x))


args.model="DAE"
DAE= model(args, train, train_n, test, test_faulty)
#DAE.optimization()
if  TO_TRAIN:
    DAE.train_model()
if  TO_TEST:
    DAE.reconstruct(train,train_ori, description="train")
    DAE.reconstruct(test,test_ori, description="test")

    z,x=DAE.reconstruct(test_faulty,test_ori ,description=args.failure)
    print(f"MSE for {args.model}:  {MSE(z, x)}")
    print(f"RR for {args.model}:  {RR(z, x)}")
    print(f"MAE for {args.model}:  {MAE(z, x)}")
    print(f"MAPE for {args.model}:  {MAPE(z, x)}")

    MSE_.append(MSE(z, x))
    RR_.append(RR(z, x))
    MAE_.append(MAE(z, x))
    MAPE_.append(MAPE(z, x))

args.model="VAE"
VAE= model(args, train, train_n, test, test_faulty)
#VAE.optimization()
if    TO_TRAIN:
    VAE.train_model()
if   TO_TEST:
    VAE.reconstruct(train,train_ori ,description="train")
    VAE.reconstruct(test,test_ori, description="test")

    z,x=VAE.reconstruct(test_faulty,test_ori ,description=args.failure)
    print(f"MSE for {args.model}:  {MSE(z, x)}")
    print(f"RR for {args.model}:  {RR(z, x)}")
    print(f"MAE for {args.model}:  {MAE(z, x)}")
    print(f"MAPE for {args.model}:  {MAPE(z, x)}")

    MSE_.append(MSE(z, x))
    RR_.append(RR(z, x))
    MAE_.append(MAE(z, x))
    MAPE_.append(MAPE(z, x))

args.model="MVAE"
MVAE= model(args, train, train_n, test, test_faulty)
#MVAE.optimization()
if  TO_TRAIN:
    MVAE.train_model()
if  TO_TEST:
    MVAE.reconstruct(train,train_ori, description="train")
    MVAE.reconstruct(test, test_ori, description="test")

    z,x=MVAE.reconstruct(test_faulty,test_ori ,description=args.failure)
    print(f"MSE for {args.model}:  {MSE(z, x)}")
    print(f"RR for {args.model}:  {RR(z, x)}")
    print(f"MAE for {args.model}:  {MAE(z, x)}")
    print(f"MAPE for {args.model}:  {MAPE(z, x)}")

    MSE_.append(MSE(z, x))
    RR_.append(RR(z, x))
    MAE_.append(MAE(z, x))
    MAPE_.append(MAPE(z, x))

if TO_TEST:
    df = pd.DataFrame({
        'Models': ['Integrated','MemAE','AE','DAE','VAE','MVAE'],
        'MSE': MSE_,
        'RR': RR_,
        'MAE': MAE_,
        'MAPE': MAPE_
    })


    source = ColumnDataSource(df)

    columns = [
        TableColumn(field='Models', title='Models'),
        TableColumn(field='MSE', title='MSE'),
        TableColumn(field='RR', title='RR'),
        TableColumn(field='MAE', title='MAE'),
        TableColumn(field='MAPE', title='MAPE')
    ]

    myTable = DataTable(source=source, columns=columns)

    show(myTable)