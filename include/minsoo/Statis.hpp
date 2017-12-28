#ifndef _STATIS_H_
#define _STATIS_H_



typedef unsigned int uint;


template <typename Dtype>
class ConvChannel {
    public:
        ConvChannel(int id, uint size);
        ~ConvChannel();
        bool addVal(Dtype value);   //append to end of list
        void printAll(void);
        void fwriteAll(std::ofstream* file);
        void fwriteBare(std::ofstream* file);
        void fwriteSimple(std::ofstream* file);
    private:
        int id;
        uint size;
        Dtype *values;
        bool *valid;
        uint pos;
};


template <typename Dtype>
class ConvLayer {
    public:
        ConvLayer(int id, uint max_channel);
        ~ConvLayer();
        bool addChannel(ConvChannel<Dtype>* channel);
        void printAll(void);
        void fwriteAll(std::ofstream* file);
        void fwriteBare(std::ofstream* file);
        void fwriteSimple(std::ofstream* file);

    private:
        int id;
        uint max_channel;
        ConvChannel<Dtype>* *channels;
        bool *valid;
        uint pos;
};

template <typename Dtype>
class IPLayer {
    public:
        IPLayer(int id, uint max_values);
        ~IPLayer();
        bool addVal(Dtype value);
        void printAll(void);
        void fwriteAll(std::ofstream* file);
        void fwriteBare(std::ofstream* file);
        void fwriteSimple(std::ofstream* file);

    private:
        int id;
        uint size;
        Dtype *values;
        bool *valid;
        uint pos;
};




template <typename Dtype> 
class Infer {
    public:
        Infer(int id, uint max_clayer, uint max_ilayer);
        ~Infer();
        bool storeResult(bool result);
        bool reportResult(void);
        bool addConvLayer(ConvLayer<Dtype>* clayer);
        uint reportCPos(void);
        bool addIPLayer(IPLayer<Dtype>* ilayer);
        uint reportIPos(void);
        int  reportID(void);
        void printAll(void);
        void fwriteAll(std::ofstream* file);
        void fwriteBare(std::ofstream* file);
        void fwriteSimple(std::ofstream* file);

    private:
        int id;
        bool evaluated; // correct value valid
        bool correct;   // 0 incorrect, 1 correct
        uint max_clayer;
        uint max_ilayer;
        ConvLayer<Dtype>* *clayers;
        IPLayer<Dtype>* *ilayers;
        bool *cvalid;
        bool *ivalid;
        uint cpos;
        uint ipos;
};


template <typename Dtype>
class Batch {
    public:
        Batch(int id, uint max_infer);
        ~Batch();
        bool addInfer(Infer<Dtype>* infer);
        Infer<Dtype>* returnInfer(uint idx);
        uint reportPos(void);
        uint reportNumCorrect(void);
        int  reportID(void);
        void printAll(void);
        void fwriteAll(std::ofstream* file);
        void fwriteBare(std::ofstream* file);
        void fwriteSimple(std::ofstream* file);

    private:
        int id;
        uint max_infer;
        Infer<Dtype>* *infers;
        bool *valid;
        uint pos;
};




template <typename Dtype> 
class Statis {
    public:
        Statis(int id, uint max_batch);
        ~Statis();
        bool addBatch(Batch<Dtype>* batch);
        Batch<Dtype>* returnBatch(uint idx);
        uint reportPos(void);
        uint reportNumCorrect(void);
        uint reportTotalInfer(void);
        float reportAccuracy(void);
        void printAll(void);
        void fwriteAll(const char* filename);
        void fwriteBare(const char* filename);
        void fwriteSimple(const char* filename);

    private:
        int id;
        uint max_batch;
        Batch<Dtype>* *batches;
        bool *valid;
        uint pos;
};




#include "Statis.cpp"

#endif
