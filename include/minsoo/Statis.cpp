#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "Statis.hpp"



template <typename Dtype>
ConvChannel<Dtype>::ConvChannel(int id, uint size) {
    this->id = id;
    this->size = size;
    this->values = new Dtype[size];
    this->valid = (bool*) calloc(size, sizeof(bool));
    this->pos = 0;

}


template <typename Dtype>
ConvChannel<Dtype>::~ConvChannel(void) {
    delete this->values;
    delete this->valid;
}

// Append to end of list
template <typename Dtype>
bool ConvChannel<Dtype>::addVal(Dtype value) {
    if (pos == size) {
        perror("ERR: ConvChannel::addVal fail: list full");
        exit(EXIT_FAILURE);
    } else if (pos > size) {
        perror("ERR: ConvChannel::addVal fail: size smaller than pos");
        exit(EXIT_FAILURE);
    }

    if (this->valid[pos]) {
        perror("ERR: ConvChannel::addVal fail: valid already set");
        exit(EXIT_FAILURE);
    }

    this->valid[pos] = true;
    this->values[pos] = value;
    this->pos++;
    return true;
}




template <typename Dtype>
void ConvChannel<Dtype>::printAll(void) {
    int i;

    std::cout << "Printing ConvChannel ID: " << this->id << std::endl;
    std::cout << "Size: " << this->size << std::endl;
    std::cout << "Position: " << this->pos << std::endl;

    for (i = 0; i < this->pos; i++) {

        std::cout << this->values[i] << std::endl;
    }
}


template <typename Dtype>
void ConvChannel<Dtype>::fwriteAll(std::ofstream* file) {
    
    if (!file) {
        perror("ERR: ConvChannel::fwriteAll failed: file NULL");
        exit(EXIT_FAILURE);
    }

    int i;

    (*file) << "Printing ConvChannel ID: " << this->id << std::endl;
    (*file) << "Size: " << this->size << std::endl;
    (*file) << "Position: " << this->pos << std::endl;

    for (i = 0; i < this->pos; i++) {

        (*file) << this->values[i] << std::endl;
    }
}


template <typename Dtype>
void ConvChannel<Dtype>::fwriteBare(std::ofstream* file) {
    
    if (!file) {
        perror("ERR: ConvChannel::fwriteBare failed: file NULL");
        exit(EXIT_FAILURE);
    }

    int i;

    (*file) << "cc " << this->id << std::endl;
    (*file) << "pos " << this->pos << std::endl;

    for (i = 0; i < this->pos; i++) {

        (*file) << this->values[i] << std::endl;
    }
}

template <typename Dtype>
void ConvChannel<Dtype>::fwriteSimple(std::ofstream* file) {
    
    if (!file) {
        perror("ERR: ConvChannel::fwriteSimple failed: file NULL");
        exit(EXIT_FAILURE);
    }

    (*file) << "Printing ConvChannel ID: " << this->id << std::endl;
    (*file) << "Size: " << this->size << std::endl;
    (*file) << "Position: " << this->pos << std::endl;
}





template <typename Dtype>
ConvLayer<Dtype>::ConvLayer(int id, uint max_channel) {
    this->id = id;
    this->max_channel = max_channel;
    this->channels = new ConvChannel<Dtype>*[max_channel];
    this->valid = (bool*) calloc(max_channel, sizeof(bool));
    this->pos = 0;
}

template <typename Dtype>
ConvLayer<Dtype>::~ConvLayer(void) {
    for (int i = 0; i < this->pos; i++) {
        delete this->channels[i];
    }
    delete this->channels;
    delete this->valid;
}

template <typename Dtype>
bool ConvLayer<Dtype>::addChannel(ConvChannel<Dtype>* channel) {
    if (pos == max_channel) {
        perror("ERR: ConvLayer::addChannel fail: list full");
        exit(EXIT_FAILURE);
    }

    if (pos > max_channel) {
        perror("ERR: ConvLayer::addChannel fail: max_channel < pos");
        exit(EXIT_FAILURE);
    }

    if (this->valid[pos]) {
        perror("ERR: ConvLayer::addChannel fail: valid already set");
        exit(EXIT_FAILURE);
    }
    if (!channel) {
        perror("ERR: ConvLayer::addChannel fail: NULL Channel");
        exit(EXIT_FAILURE);
    }

    this->valid[pos] = true;
    this->channels[pos] = channel;
    this->pos++;
    return true;
}


template <typename Dtype>
void ConvLayer<Dtype>::printAll(void) {
    int i;

    std::cout << "Printing ConvLayer ID: " << this->id << std::endl;
    std::cout << "Max_Channel: " << (int)this->max_channel << std::endl;
    std::cout << "Position: " << this->pos << std::endl;
    std::cout << std::endl;

    for (i = 0; i < this->pos; i++) {
        this->channels[i]->printAll();
        std::cout << std::endl;
    }
}


template <typename Dtype>
void ConvLayer<Dtype>::fwriteAll(std::ofstream* file) {
    int i;

    if (!file) {
        perror("ERR: ConvLayer::fwriteAll failed: file NULL");
        exit(EXIT_FAILURE);
    }

    (*file) << "Printing ConvLayer ID: " << this->id << std::endl;
    (*file) << "Max_Channel: " << (int)this->max_channel << std::endl;
    (*file) << "Position: " << this->pos << std::endl;
    (*file) << std::endl;

    for (i = 0; i < this->pos; i++) {
        this->channels[i]->fwriteAll(file);
        (*file) << std::endl;
    }
}

template <typename Dtype>
void ConvLayer<Dtype>::fwriteBare(std::ofstream* file) {
    int i;

    if (!file) {
        perror("ERR: ConvLayer::fwriteBare failed: file NULL");
        exit(EXIT_FAILURE);
    }

    (*file) << "cl " << this->id << std::endl;
    (*file) << "pos " << this->pos << std::endl;

    for (i = 0; i < this->pos; i++) {
        this->channels[i]->fwriteBare(file);
    }
}

template <typename Dtype>
void ConvLayer<Dtype>::fwriteSimple(std::ofstream* file) {
    int i;

    if (!file) {
        perror("ERR: ConvLayer::fwriteSimple failed: file NULL");
        exit(EXIT_FAILURE);
    }

    (*file) << "Printing ConvLayer ID: " << this->id << std::endl;
    (*file) << "Max_Channel: " << (int)this->max_channel << std::endl;
    (*file) << "Position: " << this->pos << std::endl;
    (*file) << std::endl;

    for (i = 0; i < this->pos; i++) {
        this->channels[i]->fwriteSimple(file);
        (*file) << std::endl;
    }
}



template <typename Dtype>
IPLayer<Dtype>::IPLayer(int id, uint size) {
    this->id = id;
    this->size = size;
    this->values = new Dtype[size];
    this->valid = (bool*) calloc(size, sizeof(bool));
    this->pos = 0;

}


template <typename Dtype>
IPLayer<Dtype>::~IPLayer(void) {
    delete this->values;
    delete this->valid;
}

// Append to end of list
template <typename Dtype>
bool IPLayer<Dtype>::addVal(Dtype value) {
    if (pos == size) {
        perror("ERR: IPLayer::addVal fail: list full");
        exit(EXIT_FAILURE);
    } else if (pos > size) {
        perror("ERR: IPLayer::addVal fail: size smaller than pos");
        exit(EXIT_FAILURE);
    }

    if (this->valid[pos]) {
        perror("ERR: IPLayer::addVal fail: valid already set");
        exit(EXIT_FAILURE);
    }

    this->valid[pos] = true;
    this->values[pos] = value;
    this->pos++;
    return true;
}




template <typename Dtype>
void IPLayer<Dtype>::printAll(void) {
    int i;

    std::cout << "Printing IPLayer ID: " << this->id << std::endl;
    std::cout << "Size: " << this->size << std::endl;
    std::cout << "Position: " << this->pos << std::endl;

    for (i = 0; i < this->pos; i++) {
        std::cout << this->values[i] << std::endl;
    }
}


template <typename Dtype>
void IPLayer<Dtype>::fwriteAll(std::ofstream* file) {
    if (!file) {
        perror("ERR: IPLayer::fwriteAll failed: file NULL");
        exit(EXIT_FAILURE);
    }

    int i;

    (*file) << "Printing IPLayer ID: " << this->id << std::endl;
    (*file) << "Size: " << this->size << std::endl;
    (*file) << "Position: " << this->pos << std::endl;

    for (i = 0; i < this->pos; i++) {

        (*file) << this->values[i] << std::endl;
    }
}

template <typename Dtype>
void IPLayer<Dtype>::fwriteBare(std::ofstream* file) {
    if (!file) {
        perror("ERR: IPLayer::fwriteBare failed: file NULL");
        exit(EXIT_FAILURE);
    }

    int i;

    (*file) << "il " << this->id << std::endl;
    (*file) << "pos " << this->pos << std::endl;

    for (i = 0; i < this->pos; i++) {

        (*file) << this->values[i] << std::endl;
    }
}


template <typename Dtype>
void IPLayer<Dtype>::fwriteSimple(std::ofstream* file) {
    
    if (!file) {
        perror("ERR: IPLayer::fwriteSimple failed: file NULL");
        exit(EXIT_FAILURE);
    }

    (*file) << "Printing IPLayer ID: " << this->id << std::endl;
    (*file) << "Size: " << this->size << std::endl;
    (*file) << "Position: " << this->pos << std::endl;
}






template <typename Dtype>
Infer<Dtype>::Infer(int id, uint max_clayer, uint max_ilayer) {
    this->id = id;
    this->evaluated = false;
    this->correct = false;
    this->correct5 = false;
    this->ref = -1;
    for (int i = 0; i < 5; i++) {
        this->pred[i] = -1;
    }
    this->max_clayer = max_clayer;
    this->max_ilayer = max_ilayer;
    this->clayers = new ConvLayer<Dtype>*[max_clayer];
    this->ilayers = new IPLayer<Dtype>*[max_ilayer];
    this->cvalid = (bool*) calloc(max_clayer, sizeof(bool));
    this->ivalid = (bool*) calloc(max_clayer, sizeof(bool));
    this->cpos = 0;
    this->ipos = 0;
}

template <typename Dtype>
Infer<Dtype>::~Infer(void) {
    for (int i = 0; i < this->cpos; i++) {
        delete this->clayers[i];
    }
    for (int i = 0; i < this->ipos; i++) {
        delete this->ilayers[i];
    }
    delete this->clayers;
    delete this->ilayers;
    delete this->cvalid;
    delete this->ivalid;
}


template <typename Dtype>
bool Infer<Dtype>::storeResult(bool result) {
    if (this->evaluated) {
        perror("ERR: Infer::evaluated fail: already evaluated");
        exit(EXIT_FAILURE);
    }

    this->evaluated = true;
    this->correct = result;
    return true;
}


template <typename Dtype>
bool Infer<Dtype>::storeResult5(bool result) {
    if (result) {
        this->correct5 = true;
        return true;
    }
    else {
        return false;
    }
}

template <typename Dtype>
bool Infer<Dtype>::storeRef(int ref) {
    this->ref = ref;
    return true;
}

template <typename Dtype>
bool Infer<Dtype>::storePred(int pred1, int pred2, int pred3, int pred4, int pred5) {
    this->pred[0] = pred1;
    this->pred[1] = pred2;
    this->pred[2] = pred3;
    this->pred[3] = pred4;
    this->pred[4] = pred5;
    return true;
}

template <typename Dtype>
bool Infer<Dtype>::reportResult(void) {
    if (!this->evaluated) {
        perror("ERR: Infer::reportResult fail: not evaluated");
        exit(EXIT_FAILURE);
    }

    return correct;
}

template <typename Dtype>
bool Infer<Dtype>::reportResult5(void) {
    return correct5;
}



template <typename Dtype>
bool Infer<Dtype>::addConvLayer(ConvLayer<Dtype>* clayer) {
    if (this->cpos == this->max_clayer) {
        perror("ERR: Infer::addConvLayer fail: list full");
        exit(EXIT_FAILURE);
    } else if (this->cpos > this->max_clayer) {
        perror("ERR: Infer::addConvLayer fail: max_clayer < cpos");
        exit(EXIT_FAILURE);
    } 
    
    if (this->cvalid[this->cpos]) {
        perror("ERR: Infer::addConvLayer fail: cvalid already set");
        exit(EXIT_FAILURE);
    }

    if (!clayer) {
        perror("ERR: Infer::addConvLayer fail: NULL Channel");
        exit(EXIT_FAILURE);
    }

    this->cvalid[this->cpos] = true;
    this->clayers[this->cpos] = clayer;
    this->cpos++;
    return true;
}

template <typename Dtype>
uint Infer<Dtype>::reportCPos(void) {
    return this->cpos;
}



template <typename Dtype>
bool Infer<Dtype>::addIPLayer(IPLayer<Dtype>* ilayer) {
    if (this->ipos == this->max_ilayer) {
        perror("ERR: Infer::addIPLayer fail: list full");
        exit(EXIT_FAILURE);
    } else if (this->ipos > this->max_ilayer) {
        perror("ERR: Infer::addIPLayer fail: max_ilayer < ipos");
        exit(EXIT_FAILURE);
    } 
    
    if (this->ivalid[this->ipos]) {
        perror("ERR: Infer::addIPLayer fail: ivalid already set");
        exit(EXIT_FAILURE);
    }

    if (!ilayer) {
        perror("ERR: Infer::addIPLayer fail: NULL Channel");
        exit(EXIT_FAILURE);
    }

    this->ivalid[this->ipos] = true;
    this->ilayers[this->ipos] = ilayer;
    this->ipos++;
    return true;
}

template <typename Dtype>
uint Infer<Dtype>::reportIPos(void) {
    return this->ipos;
}


template <typename Dtype>
int Infer<Dtype>::reportID(void) {
    return this->id;
}


template <typename Dtype>
void Infer<Dtype>::printAll(void) {
    int i;

    std::cout << "Printing Infer ID: " << this->id << std::endl;
    std::cout << "evaluated: " << this->evaluated << std::endl;
    std::cout << "correct: " << this->correct << std::endl;
    std::cout << "correct5: " << this->correct5 << std::endl;
    std::cout << "ref: " << this->ref << std::endl;
    std::cout << "pred1: " << this->pred[0] << std::endl;
    std::cout << "pred2: " << this->pred[1] << std::endl;
    std::cout << "pred3: " << this->pred[2] << std::endl;
    std::cout << "pred4: " << this->pred[3] << std::endl;
    std::cout << "pred5: " << this->pred[4] << std::endl;
    std::cout << "Max_CLayer: " << (int)this->max_clayer << std::endl;
    std::cout << "Max_ILayer: " << (int)this->max_ilayer << std::endl;
    std::cout << "Cpos: " << this->cpos << std::endl;
    std::cout << "Ipos: " << this->ipos << std::endl;
    std::cout << std::endl;

    for (i = 0; i < this->cpos; i++) {
        this->clayers[i]->printAll();
        std::cout << std::endl;
    }
    
    for (i = 0; i < this->ipos; i++) {
        this->ilayers[i]->printAll();
        std::cout << std::endl;
    }
}


template <typename Dtype>
void Infer<Dtype>::fwriteAll(std::ofstream* file) {
    int i;

    if (!file) {
        perror("ERR: Infer::fwriteAll failed: file NULL");
        exit(EXIT_FAILURE);
    }

    (*file) << "Printing Infer ID: " << this->id << std::endl;
    (*file) << "evaluated: " << this->evaluated << std::endl;
    (*file) << "correct: " << this->correct << std::endl;
    (*file) << "correct5: " << this->correct5 << std::endl;
    (*file) << "ref: " << this->ref << std::endl;
    (*file) << "pred1: " << this->pred[0] << std::endl;
    (*file) << "pred2: " << this->pred[1] << std::endl;
    (*file) << "pred3: " << this->pred[2] << std::endl;
    (*file) << "pred4: " << this->pred[3] << std::endl;
    (*file) << "pred5: " << this->pred[4] << std::endl;
    (*file) << "Max_CLayer: " << (int)this->max_clayer << std::endl;
    (*file) << "Max_ILayer: " << (int)this->max_ilayer << std::endl;
    (*file) << "Cpos: " << this->cpos << std::endl;
    (*file) << "Ipos: " << this->ipos << std::endl;
    (*file) << std::endl;

    for (i = 0; i < this->cpos; i++) {
        this->clayers[i]->fwriteAll(file);
        (*file) << std::endl;
    }
    for (i = 0; i < this->ipos; i++) {
        this->ilayers[i]->fwriteAll(file);
        (*file) << std::endl;
    }
}

template <typename Dtype>
void Infer<Dtype>::fwriteBare(std::ofstream* file) {
    int i;

    if (!file) {
        perror("ERR: Infer::fwriteBare failed: file NULL");
        exit(EXIT_FAILURE);
    }

    (*file) << "inf " << this->id << std::endl;
    (*file) << "cor " << this->correct << std::endl;
    (*file) << "cor5 " << this->correct5 << std::endl;
    (*file) << "ref: " << this->ref << std::endl;
    (*file) << "pred1: " << this->pred[0] << std::endl;
    (*file) << "pred2: " << this->pred[1] << std::endl;
    (*file) << "pred3: " << this->pred[2] << std::endl;
    (*file) << "pred4: " << this->pred[3] << std::endl;
    (*file) << "pred5: " << this->pred[4] << std::endl;
    (*file) << "cpos " << this->cpos << std::endl;
    (*file) << "ipos " << this->ipos << std::endl;

    for (i = 0; i < this->cpos; i++) {
        this->clayers[i]->fwriteBare(file);
    }
    for (i = 0; i < this->ipos; i++) {
        this->ilayers[i]->fwriteBare(file);
    }
}

template <typename Dtype>
void Infer<Dtype>::fwriteSimple(std::ofstream* file) {
    int i;

    if (!file) {
        perror("ERR: Infer::fwriteSimple failed: file NULL");
        exit(EXIT_FAILURE);
    }

    (*file) << "Printing Infer ID: " << this->id << std::endl;
    (*file) << "evaluated: " << this->evaluated << std::endl;
    (*file) << "correct: " << this->correct << std::endl;
    (*file) << "correct5: " << this->correct5 << std::endl;
    (*file) << "ref: " << this->ref << std::endl;
    (*file) << "pred1: " << this->pred[0] << std::endl;
    (*file) << "pred2: " << this->pred[1] << std::endl;
    (*file) << "pred3: " << this->pred[2] << std::endl;
    (*file) << "pred4: " << this->pred[3] << std::endl;
    (*file) << "pred5: " << this->pred[4] << std::endl;
    (*file) << "Max_CLayer: " << (int)this->max_clayer << std::endl;
    (*file) << "Max_ILayer: " << (int)this->max_ilayer << std::endl;
    (*file) << "Cpos: " << this->cpos << std::endl;
    (*file) << "Ipos: " << this->ipos << std::endl;
    (*file) << std::endl;

    for (i = 0; i < this->cpos; i++) {
        this->clayers[i]->fwriteSimple(file);
        (*file) << std::endl;
    }
    for (i = 0; i < this->ipos; i++) {
        this->ilayers[i]->fwriteSimple(file);
        (*file) << std::endl;
    }
}




template <typename Dtype>
Batch<Dtype>::Batch(int id, uint max_infer) {
    this->id = id;
    this->max_infer = max_infer;
    this->infers = new Infer<Dtype>*[max_infer];
    this->valid = (bool*) calloc(max_infer, sizeof(bool));
    this->pos = 0;
}

template <typename Dtype>
Batch<Dtype>::~Batch(void) {
    for (int i = 0; i < this->pos; i++) {
        delete this->infers[i];
    }
    delete this->infers;
    delete this->valid;
}

template <typename Dtype>
bool Batch<Dtype>::addInfer(Infer<Dtype>* infer) {
    if (pos == max_infer) {
        perror("ERR: Batch::addInfer fail: list full");
        exit(EXIT_FAILURE);
    } else if (pos > max_infer) {
        perror("ERR: Batch::addInfer fail: max_infer < pos");
        exit(EXIT_FAILURE);
    }

    if (this->valid[pos]) {
        perror("ERR: Batch::addInfer fail: valid already set");
        exit(EXIT_FAILURE);
    }
    if (!infer) {
        perror("ERR: Batch::addInfer fail: NULL Channel");
        exit(EXIT_FAILURE);
    }

    this->valid[pos] = true;
    this->infers[pos] = infer;
    this->pos++;
    return true;
}

template <typename Dtype>
Infer<Dtype>* Batch<Dtype>::returnInfer(uint idx) {
    if (idx >= this->pos) {
        perror("ERR: Batch::returnInfer: idx out of range");
        exit(EXIT_FAILURE);
    } else {
        return this->infers[idx];
    }
}


template <typename Dtype>
uint Batch<Dtype>::reportPos(void) {
    return this->pos;
}

template <typename Dtype>
uint Batch<Dtype>::reportNumCorrect(void) {
    uint num_correct = 0;

    for (int i = 0; i < this->pos; i++) {
        if (this->infers[i]->reportResult()) {
            num_correct++;
        } 
    }
    return num_correct;
}

template <typename Dtype>
uint Batch<Dtype>::reportNumCorrect5(void) {
    uint num_correct5 = 0;

    for (int i = 0; i < this->pos; i++) {
        if (this->infers[i]->reportResult5()) {
            num_correct5++;
        } 
    }
    return num_correct5;
}

template <typename Dtype>
int Batch<Dtype>::reportID(void) {
    return this->id;
}

template <typename Dtype>
void Batch<Dtype>::printAll(void) {
    int i;

    std::cout << "Printing Batch ID: " << this->id << std::endl;
    std::cout << "Max_Infer: " << (int)this->max_infer << std::endl;
    std::cout << "Position: " << this->pos << std::endl;
    std::cout << "NumCorrect: " << this->reportNumCorrect() << std::endl;
    std::cout << "NumCorrect5: " << this->reportNumCorrect5() << std::endl;
    std::cout << std::endl;

    for (i = 0; i < this->pos; i++) {
        this->infers[i]->printAll();
        std::cout << std::endl;
    }
}


template <typename Dtype>
void Batch<Dtype>::fwriteAll(std::ofstream* file) {
    int i;

    if (!file) {
        perror("ERR: Batch::fwriteAll failed: file NULL");
        exit(EXIT_FAILURE);
    }

    (*file) << "Printing Batch ID: " << this->id << std::endl;
    (*file) << "Max_Infer: " << (int)this->max_infer << std::endl;
    (*file) << "Position: " << this->pos << std::endl;
    (*file) << "NumCorrect: " << this->reportNumCorrect() << std::endl;
    (*file) << "NumCorrect5: " << this->reportNumCorrect5() << std::endl;
    (*file) << std::endl;

    for (i = 0; i < this->pos; i++) {
        this->infers[i]->fwriteAll(file);
        (*file) << std::endl;
    }
}


template <typename Dtype>
void Batch<Dtype>::fwriteBare(std::ofstream* file) {
    int i;

    if (!file) {
        perror("ERR: Batch::fwriteBare failed: file NULL");
        exit(EXIT_FAILURE);
    }

    (*file) << "bat " << this->id << std::endl;
    (*file) << "pos " << this->pos << std::endl;

    for (i = 0; i < this->pos; i++) {
        this->infers[i]->fwriteBare(file);
    }
}

template <typename Dtype>
void Batch<Dtype>::fwriteBareOne(std::ofstream* file, int infnum) {
    int i;

    if (!file) {
        perror("ERR: Batch::fwriteBareOne failed: file NULL");
        exit(EXIT_FAILURE);
    }

    (*file) << "bat " << this->id << std::endl;
    (*file) << "pos " << this->pos << std::endl;

    if (infnum < 0) {
        perror("ERR: Batch:fwriteBareOne failed: infnum < 0");
        exit(EXIT_FAILURE);
    } else if (infnum >= this->pos) {
        perror("ERR: Batch:fwriteBareOne failed: infnum >= num_inf");
        exit(EXIT_FAILURE);
    } else {
        this->infers[infnum]->fwriteBare(file);
    }

}

template <typename Dtype>
void Batch<Dtype>::fwriteBareSel(std::ofstream* file, int infnum[], int size) {
    int i;

    if (!file) {
        perror("ERR: Batch::fwriteBareSel failed: file NULL");
        exit(EXIT_FAILURE);
    }

    (*file) << "bat " << this->id << std::endl;
    (*file) << "pos " << this->pos << std::endl;
    
    for (i = 0; i < size; i++) {
        int curr_num = infnum[i];
        
        if (curr_num < 0) {
            perror("ERR: Batch:fwriteBareSel failed: curr_num < 0");
            exit(EXIT_FAILURE);
        } else if (curr_num >= this->pos) {
            perror("ERR: Batch:fwriteBareSel failed: curr_num >= num_inf");
            exit(EXIT_FAILURE);
        } else {
            this->infers[curr_num]->fwriteBare(file);
        }
    }
}

template <typename Dtype>
void Batch<Dtype>::fwriteSimple(std::ofstream* file) {
    int i;

    if (!file) {
        perror("ERR: Batch::fwriteSimple failed: file NULL");
        exit(EXIT_FAILURE);
    }

    (*file) << "Printing Batch ID: " << this->id << std::endl;
    (*file) << "Max_Infer: " << (int)this->max_infer << std::endl;
    (*file) << "Position: " << this->pos << std::endl;
    (*file) << "NumCorrect: " << this->reportNumCorrect() << std::endl;
    (*file) << "NumCorrect5: " << this->reportNumCorrect5() << std::endl;
    (*file) << std::endl;

    for (i = 0; i < this->pos; i++) {
        this->infers[i]->fwriteSimple(file);
        (*file) << std::endl;
    }
}






template <typename Dtype>
Statis<Dtype>::Statis(int id, uint max_batch) {
    this->id = id;
    this->max_batch = max_batch;
    this->batches = new Batch<Dtype>*[max_batch];
    this->valid = (bool*) calloc(max_batch, sizeof(bool));
    this->pos = 0;
}

template <typename Dtype>
Statis<Dtype>::~Statis(void) {
    for (int i = 0; i < this->pos; i++) {
        delete this->batches[i];
    }
    delete this->batches;
    delete this->valid;
}

/*
template <typename Dtype>
bool Statis<Dtype>::turn_on(void) {
    if (this->on_off) {
        perror("ERR: Statis::turn_on fail: already on");
        exit(EXIT_FAILURE);
    }

    this->on_off = true;
    return true;
}

template <typename Dtype>
bool Statis<Dtype>::turn_off(void) {
    if (!this->on_off) {
        perror("ERR: Statis::turn_off fail: already off");
        exit(EXIT_FAILURE);
    }

    this->on_off = false;
    return true;
}


template <typename Dtype>
bool Statis<Dtype>::reportOn(void) {
    return this->on_off;
}


template <typename Dtype>
int Statis<Dtype>::reportNumCorrect(void) {
    return this->num_correct;
}



template <typename Dtype>
float Statis<Dtype>::reportAccuracy(void) {
    return ((float)this->num_correct)/((float)this->pos);
}

*/

template <typename Dtype>
bool Statis<Dtype>::addBatch(Batch<Dtype>* batch) {
    if (this->pos == this->max_batch) {
        perror("ERR: Statis::addBatch fail: list full");
        exit(EXIT_FAILURE);
    } else if (this->pos > this->max_batch) {
        perror("ERR: Statis::addBatch fail: max_batch < pos");
        exit(EXIT_FAILURE);
    } 
    
    if (this->valid[this->pos]) {
        perror("ERR: Statis::addBatch fail: valid already set");
        exit(EXIT_FAILURE);
    }

    if (!batch) {
        perror("ERR: Statis::addBatch fail: NULL Channel");
        exit(EXIT_FAILURE);
    }

    this->valid[this->pos] = true;
    this->batches[this->pos] = batch;
    this->pos++;
    
    return true;
}

template <typename Dtype>
Batch<Dtype>* Statis<Dtype>::returnBatch(uint idx) {
    if (idx >= this->pos) {
        perror("ERR: Statis::returnBatch: idx out of range");
        return NULL;
    } else {
        return this->batches[idx];
    }
}



template <typename Dtype>
uint Statis<Dtype>::reportPos(void) {
    return this->pos;
}



template <typename Dtype>
uint Statis<Dtype>::reportNumCorrect(void) {
    uint num_correct = 0;

    for (int i = 0; i < this->pos; i++) {
        num_correct += this->batches[i]->reportNumCorrect();    
    }

    return num_correct;
}

template <typename Dtype>
uint Statis<Dtype>::reportNumCorrect5(void) {
    uint num_correct5 = 0;

    for (int i = 0; i < this->pos; i++) {
        num_correct5 += this->batches[i]->reportNumCorrect5();    
    }

    return num_correct5;
}

template <typename Dtype>
uint Statis<Dtype>::reportTotalInfer(void) {
    uint total_infer = 0;

    for (int i = 0; i < this->pos; i++) {
        total_infer += this->batches[i]->reportPos();
    }
    return total_infer;
}

template <typename Dtype>
float Statis<Dtype>::reportAccuracy(void) {
    uint num_correct = this->reportNumCorrect();
    uint total_infer = this->reportTotalInfer();

    return ((float)num_correct)/((float)total_infer);
}

template <typename Dtype>
float Statis<Dtype>::reportAccuracy5(void) {
    uint num_correct5 = this->reportNumCorrect5();
    uint total_infer = this->reportTotalInfer();

    return ((float)num_correct5)/((float)total_infer);
}

template <typename Dtype>
void Statis<Dtype>::printAll(void) {
    int i;

    std::cout << "Printing Statis ID: " << this->id << std::endl;
    std::cout << "Max_Batch: " << (int)this->max_batch << std::endl;
    std::cout << "pos: " << this->pos << std::endl;
    std::cout << "NumCorrect: " << this->reportNumCorrect() << std::endl;
    std::cout << "NumCorrect5: " << this->reportNumCorrect5() << std::endl;
    std::cout << "TotalInfer: " << this->reportTotalInfer() << std::endl;
    std::cout << "Accuracy: " << this->reportAccuracy() << std::endl;
    std::cout << "Accuracy5: " << this->reportAccuracy5() << std::endl;
    std::cout << std::endl;

    for (i = 0; i < this->pos; i++) {
        this->batches[i]->printAll();
        std::cout << std::endl;
    }
}


template <typename Dtype>
void Statis<Dtype>::fwriteAll(const char* filename) {
    int i;

    std::ofstream fout;
    fout.open(filename, std::ios::out);


    if (!fout) {
        perror("ERR: Statis::fwriteAll failed: fout NULL");
        exit(EXIT_FAILURE);
    }

    fout << "Printing Statis ID: " << this->id << std::endl;
    fout << "Max_Batch: " << (int)this->max_batch << std::endl;
    fout << "pos: " << this->pos << std::endl;
    fout << "NumCorrect: " << this->reportNumCorrect() << std::endl;
    fout << "NumCorrect5: " << this->reportNumCorrect5() << std::endl;
    fout << "TotalInfer: " << this->reportTotalInfer() << std::endl;
    fout << "Accuracy: " << this->reportAccuracy() << std::endl;
    fout << "Accuracy5: " << this->reportAccuracy5() << std::endl;
    fout << std::endl;

    for (i = 0; i < this->pos; i++) {
        this->batches[i]->fwriteAll(&fout);
        fout << std::endl;
    }

    fout.close();
}

template <typename Dtype>
void Statis<Dtype>::fwriteBare(const char* filename) {
    int i;

    std::ofstream fout;
    fout.open(filename, std::ios::out);


    if (!fout) {
        perror("ERR: Statis::fwriteBare failed: fout NULL");
        exit(EXIT_FAILURE);
    }

    fout << "stat " << this->id << std::endl;
    fout << "pos " << this->pos << std::endl;
    fout << "numcor " << this->reportNumCorrect() << std::endl;
    fout << "numcor5 " << this->reportNumCorrect5() << std::endl;
    fout << "total " << this->reportTotalInfer() << std::endl;
    fout << "acc " << this->reportAccuracy() << std::endl;
    fout << "acc5 " << this->reportAccuracy5() << std::endl;

    for (i = 0; i < this->pos; i++) {
        this->batches[i]->fwriteBare(&fout);
    }

    fout.close();
}


template <typename Dtype>
void Statis<Dtype>::fwriteBareOne(const char* filename, int batchnum, int infnum) {
    int i;

    std::ofstream fout;
    fout.open(filename, std::ios::out);


    if (!fout) {
        perror("ERR: Statis::fwriteBareOne failed: fout NULL");
        exit(EXIT_FAILURE);
    }

    fout << "stat " << this->id << std::endl;
    fout << "pos " << this->pos << std::endl;
    fout << "numcor " << this->reportNumCorrect() << std::endl;
    fout << "numcor5 " << this->reportNumCorrect5() << std::endl;
    fout << "total " << this->reportTotalInfer() << std::endl;
    fout << "acc " << this->reportAccuracy() << std::endl;
    fout << "acc5 " << this->reportAccuracy5() << std::endl;

    if (batchnum < 0) {
        perror("ERR: Statis:fwriteBareOne failed: batchnum < 0");
        exit(EXIT_FAILURE);
    } else if (batchnum >= this->pos) {
        perror("ERR: Statis:fwriteBareOne failed: batchnum >= num_batch");
        exit(EXIT_FAILURE);
    } else {
        this->batches[batchnum]->fwriteBareOne(&fout, infnum);
    }

    fout.close();
}


template <typename Dtype>
void Statis<Dtype>::fwriteBareSel(const char* filename, int batchnum, int infnum[], int size) {

    std::ofstream fout;
    fout.open(filename, std::ios::out);


    if (!fout) {
        perror("ERR: Statis::fwriteBareSel failed: fout NULL");
        exit(EXIT_FAILURE);
    }

    fout << "stat " << this->id << std::endl;
    fout << "pos " << this->pos << std::endl;
    fout << "numcor " << this->reportNumCorrect() << std::endl;
    fout << "numcor5 " << this->reportNumCorrect5() << std::endl;
    fout << "total " << this->reportTotalInfer() << std::endl;
    fout << "acc " << this->reportAccuracy() << std::endl;
    fout << "acc5 " << this->reportAccuracy5() << std::endl;

    if (batchnum < 0) {
        perror("ERR: Statis:fwriteBareSel failed: batchnum < 0");
        exit(EXIT_FAILURE);
    } else if (batchnum >= this->pos) {
        perror("ERR: Statis:fwriteBareSel failed: batchnum >= num_batch");
        exit(EXIT_FAILURE);
    } else {
        this->batches[batchnum]->fwriteBareSel(&fout, infnum, size);
    }

    fout.close();
}



template <typename Dtype>
void Statis<Dtype>::fwriteSimple(const char* filename) {
    int i;

    std::ofstream fout;
    fout.open(filename, std::ios::out);


    if (!fout) {
        perror("ERR: Statis::fwriteSimple failed: fout NULL");
        exit(EXIT_FAILURE);
    }

    fout << "Printing Statis ID: " << this->id << std::endl;
    fout << "Max_Batch: " << (int)this->max_batch << std::endl;
    fout << "pos: " << this->pos << std::endl;
    fout << "NumCorrect: " << this->reportNumCorrect() << std::endl;
    fout << "NumCorrect5: " << this->reportNumCorrect5() << std::endl;
    fout << "TotalInfer: " << this->reportTotalInfer() << std::endl;
    fout << "Accuracy: " << this->reportAccuracy() << std::endl;
    fout << "Accuracy5: " << this->reportAccuracy5() << std::endl;
    fout << std::endl;

    for (i = 0; i < this->pos; i++) {
        this->batches[i]->fwriteSimple(&fout);
        fout << std::endl;
    }

    fout.close();
}

