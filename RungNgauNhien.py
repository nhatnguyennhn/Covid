import numpy 
import pandas 
#from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
duLieuTrain = pandas.read_csv("train.csv", index_col = 'Id')
duLieuTest = pandas.read_csv("test.csv", index_col = 'ForecastId')
pandas.set_option('display.max_columns', 150)
pandas.set_option('display.max_rows', 150)
yTrainSoTruongHop = numpy.array(duLieuTrain['ConfirmedCases'].astype(int))
yTrainSoNguoiChet = numpy.array(duLieuTrain['Fatalities'].astype(int))
cotSoLieu = ['ConfirmedCases', 'Fatalities']

duLieuChung = pandas.concat([duLieuTrain.drop(cotSoLieu, axis=1), duLieuTest])
chiSo = duLieuTrain.shape[0]
duLieuChung = pandas.get_dummies(duLieuChung, columns=duLieuChung.columns)
xTrain = duLieuChung[:chiSo]
xTest= duLieuChung[chiSo:]
TScayQuyetDinh = {'max_features':  [10, 100, 150], 'min_samples_leaf': [8, 10, 12], 'max_depth': [10, 20, 30]}
cayQuyetDinhHoiQuy = RandomForestRegressor(n_estimators=100, random_state=5, n_jobs= -1)
"""
Ở đây mình GridSearchCV để lấy hyper parameter cho thuật toán random(bởi vì tạo sẵn parameter thì nó sẽ ảnh hưởng đến độ chính xác bài toán)
 nhưng vì lý do nó quá lâu(mất gần 15 phút để nó tìm ra cho data này )
Nếu bạn muốn thì sử dụng theo cách này
gridsearch = GridSearchCV(cayQuyetDinhHoiQuy, TScayQuyetDinh, n_jobs=-1, cv=5, verbose=1)
gridsearch.fit(xTrain,yTrainSoTruongHop)
cayQuyetDinh = RandomForestRegressor(max_depth = gridsearch.best_params_['max_depth'], max_features = gridsearch.best_params_['max_features'], min_samples_leaf=gridsearch.best_params_['min_samples_leaf'], random_state=5, n_estimators=100, n_jobs= -1)
kết quả ta thu được 
max_depth =10, max_features = 100, min_samples_leaf=12 
Có 1 cách khác là sử dụng random nhưng nó không chạy đủ trường hợp nên dễ bỏ qua hypẻ parameter
"""
cayQuyetDinh = RandomForestRegressor(max_depth =10, max_features = 100, min_samples_leaf=12, random_state=5, n_estimators=100, n_jobs= -1)
cayQuyetDinh.fit(xTrain,yTrainSoTruongHop) 

yTestSoTruongHop = cayQuyetDinh.predict(xTest)
yTestSoTruongHop= yTestSoTruongHop.astype(int)
yTestSoTruongHop[yTestSoTruongHop <0]=0
#cayQuyetDinh = RandomForestRegressor(max_depth = gridsearch.best_params_['max_depth'], max_features = gridsearch.best_params_['max_features'], min_samples_leaf=gridsearch.best_params_['min_samples_leaf'], random_state=5, n_estimators=100, n_jobs= -1)
cayQuyetDinh = RandomForestRegressor(max_depth =10, max_features = 100, min_samples_leaf=12, random_state=5, n_estimators=100, n_jobs= -1)
cayQuyetDinh.fit(xTrain,yTrainSoNguoiChet) 
yTestSoNguoiChet = cayQuyetDinh.predict(xTest)
yTestSoNguoiChet= yTestSoNguoiChet.astype(int)
yTestSoNguoiChet[yTestSoNguoiChet <0]=0
ketQua = pandas.DataFrame([yTestSoTruongHop, yTestSoNguoiChet], index = ['SoTruongHop','SoNguoiChet'], columns= numpy.arange(1,yTestSoTruongHop.shape[0] + 1)).T
ketQua.to_csv('ketqua.csv', index_label = "ID")
