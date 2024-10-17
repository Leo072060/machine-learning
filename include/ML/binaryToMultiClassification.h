#include <vector>
#include <algorithm>
#include <random>
#include <memory>

#include"modelBase.h"

using namespace std;

template<class T>
class MvM_ECOC : public ClassificationModelBase<T>
{
public:
    MvM_ECOC():ECOC(this->administrator),MODELS(this->administrator),LABELS(this->administrator),LABEL_INDEXS(this->administrator){}
// * * * * * * * functions * * * * * * *
public:
	void    train  (const Mat<T>& x, const Mat<string>& y) override;
	Mat<string>  predict(const Mat<T>& x) const			   override;
    void set_binary_classification_model(const ClassificationModelBase<T>& model) { binary_classification_model = model.clone(); }
private:
    shared_ptr<ClassificationModelBase<T>>  clone() const override{ return make_shared<MvM_ECOC<T>>(*this); }
    void generate_labalIndexs(const Mat<T>& x, const Mat<string>& y);
    void generate_ECOC();
// * * * * * * * attributes * * * * * * *
public:
    string multi_class = "ovr";
private:
    shared_ptr<ClassificationModelBase<T>> binary_classification_model;
    ManagedVal<Mat<int>> ECOC;
    ManagedVal<vector<shared_ptr<ClassificationModelBase<T>>>> MODELS;
    ManagedVal<Mat<string>> LABELS;
    ManagedVal<vector<vector<size_t>>> LABEL_INDEXS;
};

#pragma member functions
template<typename T>
void MvM_ECOC<T>::train(const Mat<T>& x, const Mat<string>& y)
{
    generate_labalIndexs(x,y);
    generate_ECOC();
    vector<shared_ptr<ClassificationModelBase<T>>> models;
    for(size_t i=0; i<ECOC.read().size_column(); ++i)
    {
        vector<size_t> index_true;
        vector<size_t> index_false;
        for(size_t j =0; j<ECOC.read().size_row(); ++j)
        {
            if     (1 == ECOC.read().iloc(i,j))  index_true.insert(index_true.end(),LABEL_INDEXS.read()[j].cbegin(),LABEL_INDEXS.read()[j].cend());
            else if(-1 == ECOC.read().iloc(i,j)) index_false.insert(index_false.end(),LABEL_INDEXS.read()[j].cbegin(),LABEL_INDEXS.read()[j].cend());
        }
        vector<size_t> index(index_true);
        index.insert(index.end(),index_false.cbegin(),index_false.cend());
        Mat<T> train_x = x.extract_rows(index);
        Mat<string> train_y = y.extract_rows(index);
        for(const auto e:index_true) train_y.iloc(e,0) = "T";
        for(const auto e:index_false) train_y.iloc(e,0) = "F";
        shared_ptr<ClassificationModelBase<T>> model = binary_classification_model->clone();
        model->train(train_x,train_y);
        models.push_back(model);
    }
    this->record(MODELS,models);
}
template<typename T>
void MvM_ECOC<T>::generate_labalIndexs(const Mat<T>& x, const Mat<string>& y)
{
    Mat<string> labels = unique(y);
    if (labels.size() < 3) 
        throw std::runtime_error("Warning: Binary classification detected. This method is intended for multi-class problems. Please use a binary classifier instead.");
    labels.sort_column(0);
    vector<vector<size_t>> labelIndexs(labels.size());
    for(size_t i = 0; i < y.size_row(); ++i)
        labelIndexs[labels.find(y.iloc(i,0)).iloc(0,1)].push_back(i);
    this->record(LABELS,labels);
    this->record(LABEL_INDEXS,labelIndexs);
}
template<typename T>
void MvM_ECOC<T>::generate_ECOC()
{
    if(multi_class == "ovr")
    {
        Mat<int> ecoc(LABELS.read().size_column(),LABELS.read().size_column());
        ecoc.fill(-1);
        for(size_t i=0; i < ecoc.size_row(); ++i) 
        {
            ecoc.iloc(i,i) = 1;
            ecoc.iloc_rowName(i) = LABELS.read().iloc(0,i);
        }
        this->record(ECOC,ecoc);
    }
    else throw runtime_error("Error: Unsupported multi-class strategy.");
}
template<typename T>
Mat<string>  MvM_ECOC<T>::predict(const Mat<T>& x) const
{
    Mat<string> ret(x.size_row(),1);
    for(size_t i=0;i<x.size_row();++i)
    {
        Mat<int> pred_ecoc(1,ECOC.read().size_column());
        for(size_t j=0; j<MODELS.read().size(); ++j)
        {
            Mat<string> pred_y = MODELS.read()[j]->predict(x.iloc_row(i));
            if(pred_y.iloc(0,0) == "T") pred_ecoc.iloc(0,j) = 1;
            else if(pred_y.iloc(0,0) == "F") pred_ecoc.iloc(0,j) = -1;
        }
        int max_HammingDistance = -1;
        string pred_y;
        for(size_t j=0;j<ECOC.read().size_row(); ++j)
        {
            int HammingDistance=0;
            for(size_t k=0;k<ECOC.read().size_column(); ++k)
            {
                if(ECOC.read().iloc(j,k)!=pred_ecoc.iloc(0,k)) ++HammingDistance;
            }
            if(HammingDistance>max_HammingDistance) pred_y = ECOC.read().iloc_rowName(j);
        }
        ret.iloc(i,0) = pred_y;
    }
    return ret;
}
#pragma endregion