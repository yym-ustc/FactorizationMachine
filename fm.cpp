#pragma GCC diagnostic ignored "-Wunused-result" 
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <new>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <cstring>
#include <vector>
#include <cassert>
#include <numeric>
#include <cmath>

#if defined USEOMP
#include <omp.h>
#endif

#include "fm.h"
#include "timer.h"

namespace fm {

namespace {

using namespace std;

fm_int const kCHUNK_SIZE = 10000000;
fm_int const kMaxLineSize = 100000;

fm_float uniform() {
    return rand() / ((double)RAND_MAX + 1.0);
}

fm_float gaussian() {
    fm_float u,v, x, y, Q;
    do {
        do {
            u = uniform();
        } while (u == 0.0);
        v = 1.7156 * (uniform() - 0.5);
        x = u - 0.449871;
        y = fabs(v) + 0.386595;
        Q = x * x + y * (0.19600 * y - 0.25472 * x);
    } while (Q >= 0.27597 && (Q > 0.27846 || v * v > -4.0 * u * u * log(u)));
    return v / u;
}

double gaussian(double mean, double stdev) {
    if(0.0 == stdev) {
        return mean;
    } else {
        return mean + stdev * gaussian();
    }
}

inline fm_float wTx(
    fm_node *begin,
    fm_node *end,
    fm_model &model, 
    fm_float kappa=0, 
    fm_float eta=0, 
    fm_float lambda=0, 
    bool do_update=false) {

    fm_float res = 0;
    if (do_update) {
        // weight w0
        fm_float w0_grad = kappa + lambda * model.w0;
        model.w0_a += w0_grad * w0_grad;
        model.w0 -= eta / sqrt(model.w0_a) * w0_grad;
        for (fm_node *node1 = begin; node1 != end; node1++) { // one feature in a sample
            fm_int idx1 = node1->idx;
            fm_float value1 = node1->value;
            if (idx1 >= model.n)
                continue;
            // weight wi
            fm_weight_unit &unit1 = model.weight_map.find(idx1)->second; // weight unit of the feature 
            fm_float w_grad = kappa * value1 + lambda * unit1.w;
            unit1.w_a += w_grad * w_grad;
            unit1.w -= eta / sqrt(unit1.w_a) * w_grad;
            // latent vector i
            for (fm_int f = 0; f < model.k; f++) {
                fm_float v_grad = 0;
                for (fm_node *node2 = begin; node2 != end; node2++) {
                    fm_int idx2 = node2->idx;
                    fm_float value2 = node2->value;
                    fm_weight_unit &unit2 = model.weight_map.find(idx2)->second;
                    v_grad += unit2.v.at(f) * value2;
                }
                v_grad = value1 * v_grad - unit1.v.at(f) * value1 * value1;
                v_grad = (kappa * v_grad + lambda * unit1.v.at(f));
                unit1.v_a.at(f) += v_grad * v_grad;
                unit1.v.at(f) -= eta / sqrt(unit1.v_a.at(f)) * v_grad;
            }  
        }
    } else {
        // weight w0
        res += model.w0;
        // weight wi
        for (fm_node *node = begin; node != end; node++) {
            fm_float idx = node->idx;
            fm_float value = node->value;
            fm_weight_unit &unit = model.weight_map.find(idx)->second;
            res += unit.w * value;
        }
        // latent vector i
        fm_float latent_res = 0;
        for (fm_int f = 0; f < model.k; f++) {
            fm_float sum_square = 0;
            fm_float square_sum = 0;
            for (fm_node *node = begin; node != end; node++) {
                fm_float idx = node->idx;
                fm_float value = node->value;
                fm_weight_unit &unit = model.weight_map.find(idx)->second;
                sum_square += unit.v.at(f) * value;
                square_sum += unit.v.at(f) * unit.v.at(f) * value * value;
            }
            latent_res += (sum_square * sum_square - square_sum);
        }
        res += (latent_res * 0.5);
    }
    return res;
}

fm_model init_model(fm_int n, fm_parameter param) {
    fm_model model;
    model.n = n;
    model.k = param.k;
    model.weight_map.clear();
    model.w0 = 0;
    model.w0_a = 1;

    default_random_engine generator;
    uniform_real_distribution<fm_float> distribution(0.0, 1.0);

    for (fm_int i = 0; i < model.n; i++) {
        fm_weight_unit unit;
        unit.w = 0;
        unit.w_a = 1;
        for (fm_int j = 0; j < model.k; j++) {
            fm_float v_temp = gaussian(0, param.stdev);
            unit.v.push_back(v_temp);
            unit.v_a.push_back(1);
        }
        model.weight_map.insert(make_pair(i, unit));
    }

    return model;
}

struct disk_problem_meta {
    fm_int n = 0;
    fm_int l = 0;
    fm_int num_blocks = 0;
    fm_long B_pos = 0;
    uint64_t hash1;
    uint64_t hash2;
};

struct problem_on_disk {
    disk_problem_meta meta;
    vector<fm_float> Y;
    vector<fm_long> P;
    vector<fm_node> X;
    vector<fm_long> B;

    problem_on_disk(string path) {
        f.open(path, ios::in | ios::binary);
        if (f.good()) {
            f.read(reinterpret_cast<char*>(&meta), sizeof(disk_problem_meta));
            f.seekg(meta.B_pos);
            B.resize(meta.num_blocks);
            f.read(reinterpret_cast<char*>(B.data()), sizeof(fm_long) * meta.num_blocks);
        }
    }

    int load_block(int block_index) {
        if(block_index >= meta.num_blocks)
            assert(false);

        f.seekg(B[block_index]);

        fm_int l;
        f.read(reinterpret_cast<char*>(&l), sizeof(fm_int));

        Y.resize(l);
        f.read(reinterpret_cast<char*>(Y.data()), sizeof(fm_float) * l);

        P.resize(l+1);
        f.read(reinterpret_cast<char*>(P.data()), sizeof(fm_long) * (l+1));

        X.resize(P[l]);
        f.read(reinterpret_cast<char*>(X.data()), sizeof(fm_node) * P[l]);

        return l;
    }

    bool is_empty() {
        return meta.l == 0;
    }

private:
    ifstream f;
};

uint64_t hashfile(string txt_path, bool one_block=false) {
    ifstream f(txt_path, ios::ate | ios::binary);
    if (f.bad())
        return 0;

    fm_long end = (fm_long) f.tellg();
    f.seekg(0, ios::beg);
    assert(static_cast<int>(f.tellg()) == 0);

    uint64_t magic = 90359;
    for (fm_long pos = 0; pos < end; ) {
        fm_long next_pos = min(pos + kCHUNK_SIZE, end);
        fm_long size = next_pos - pos;
        vector<char> buffer(kCHUNK_SIZE);
        f.read(buffer.data(), size);

        fm_int i = 0;
        while (i < size - 8) {
            uint64_t x = *reinterpret_cast<uint64_t*>(buffer.data() + i);
            magic = ( (magic + x) * (magic + x + 1) >> 1) + x;
            i += 8;
        }
        for (; i < size; i++) {
            char x = buffer[i];
            magic = ( (magic + x) * (magic + x + 1) >> 1) + x;
        }

        pos = next_pos;
        if (one_block)
            break;
    }

    return magic;
}

void txt2bin(string txt_path, string bin_path) {
    
    FILE *f_txt = fopen(txt_path.c_str(), "r");
    if (f_txt == nullptr)
        throw;

    ofstream f_bin(bin_path, ios::out | ios::binary);

    vector<char> line(kMaxLineSize);

    fm_long p = 0;
    disk_problem_meta meta;

    vector<fm_float> Y;
    vector<fm_long> P(1, 0);
    vector<fm_node> X;
    vector<fm_long> B;

    auto write_chunk = [&] () {
        B.push_back(f_bin.tellp());
        fm_int l = Y.size();
        fm_long nnz = P[l];
        meta.l += l;

        f_bin.write(reinterpret_cast<char*>(&l), sizeof(fm_int));
        f_bin.write(reinterpret_cast<char*>(Y.data()), sizeof(fm_float) * l);
        f_bin.write(reinterpret_cast<char*>(P.data()), sizeof(fm_long) * (l+1));
        f_bin.write(reinterpret_cast<char*>(X.data()), sizeof(fm_node) * nnz);

        Y.clear();
        P.assign(1, 0);
        X.clear();
        p = 0;
        meta.num_blocks++;
    };

    f_bin.write(reinterpret_cast<char*>(&meta), sizeof(disk_problem_meta));

    while (fgets(line.data(), kMaxLineSize, f_txt)) {
        char *y_char = strtok(line.data(), " \t");

        fm_float y = (atoi(y_char)>0)? 1.0f : -1.0f;

        for (; ; p++) {
            char *idx_char = strtok(nullptr,":");
            char *value_char = strtok(nullptr," \t");

            if(idx_char == nullptr || *idx_char == '\n')
                break;

            fm_node N;
            N.idx = atoi(idx_char);
            N.value = atof(value_char);

            X.push_back(N);

            meta.n = max(meta.n, N.idx+1);
        }

        Y.push_back(y);
        P.push_back(p);

        if (X.size() > (size_t)kCHUNK_SIZE)
            write_chunk(); 
    }
    write_chunk(); 
    write_chunk(); // write a dummy empty chunk in order to know where the EOF is
    assert(meta.num_blocks == (fm_int)B.size());
    meta.B_pos = f_bin.tellp();
    f_bin.write(reinterpret_cast<char*>(B.data()), sizeof(fm_long) * B.size());

    fclose(f_txt);
    meta.hash1 = hashfile(txt_path, true);
    meta.hash2 = hashfile(txt_path, false);

    f_bin.seekp(0, ios::beg);
    f_bin.write(reinterpret_cast<char*>(&meta), sizeof(disk_problem_meta));
}

bool check_same_txt_bin(string txt_path, string bin_path) {
    ifstream f_bin(bin_path, ios::binary | ios::ate);
    if (f_bin.tellg() < (fm_long)sizeof(disk_problem_meta))
        return false;
    disk_problem_meta meta;
    f_bin.seekg(0, ios::beg);
    f_bin.read(reinterpret_cast<char*>(&meta), sizeof(disk_problem_meta));
    if (meta.hash1 != hashfile(txt_path, true))
        return false;
    if (meta.hash2 != hashfile(txt_path, false))
        return false;
    return true;
}

} // unnamed namespace

fm_model::~fm_model() {

}

void fm_read_problem_to_disk(string txt_path, string bin_path) {
    Timer timer;
    
    cout << "First check if the text file has already been converted to binary format " << flush;
    bool same_file = check_same_txt_bin(txt_path, bin_path);
    cout << "(" << fixed << setprecision(1) << timer.toc() << " seconds)" << endl;
    if(same_file) {
        cout << "Binary file found. Skip converting text to binary" << endl;
    } else {
        cout << "Binary file NOT found. Convert text file to binary file " << flush;
        txt2bin(txt_path, bin_path);
        cout << "(" << fixed << setprecision(1) << timer.toc() << " seconds)" << endl;
    }
}

fm_model fm_train_on_disk(string tr_path, string va_path, fm_parameter param, string model_path) {

    problem_on_disk tr(tr_path);
    problem_on_disk va(va_path);

    fm_model model = init_model(tr.meta.n, param);

    bool auto_stop = param.auto_stop && !va_path.empty();

    fm_double best_va_loss = numeric_limits<fm_double>::max();

    cout.width(4);
    cout << "iter";
    cout.width(13);
    cout << "tr_logloss";
    if (!va_path.empty()) {
        cout.width(13);
        cout << "va_logloss";
    }
    cout.width(13);
    cout << "tr_time";
    cout << endl;

    Timer timer;

    auto one_epoch = [&] (problem_on_disk &prob, bool do_update) {

        fm_double loss = 0;

        vector<fm_int> outer_order(prob.meta.num_blocks);
        iota(outer_order.begin(), outer_order.end(), 0);
        random_shuffle(outer_order.begin(), outer_order.end());
        for (auto blk : outer_order) {
            fm_int l = prob.load_block(blk);

            vector<fm_int> inner_order(l);
            iota(inner_order.begin(), inner_order.end(), 0);
            random_shuffle(inner_order.begin(), inner_order.end());

#if defined USEOMP
#pragma omp parallel for schedule(static) reduction(+: loss)
#endif
            for (fm_int ii = 0; ii < l; ii++) {
                fm_int i = inner_order[ii];

                fm_float y = prob.Y[i];
                
                fm_node *begin = &prob.X[prob.P[i]];

                fm_node *end = &prob.X[prob.P[i+1]];

                fm_double t = wTx(begin, end, model);

                fm_double expnyt = exp(-y*t);

                loss += log1p(expnyt);

                if(do_update) {
                   
                    fm_float kappa = -y*expnyt/(1+expnyt);
                    wTx(begin, end, model, kappa, param.eta, param.lambda, true);
                }
            }
        }

        return loss / prob.meta.l;
    };

    for (fm_int iter = 1; iter <= param.nr_iters; iter++) {
        timer.tic();
        fm_double tr_loss = one_epoch(tr, true);
        timer.toc();

        cout.width(4);
        cout << iter;
        cout.width(13);
        cout << fixed << setprecision(5) << tr_loss;

        if (!va.is_empty()) {
            fm_double va_loss = one_epoch(va, false);

            cout.width(13);
            cout << fixed << setprecision(5) << va_loss;

            if (auto_stop) {
                if(va_loss > best_va_loss) {
                    cout << endl << "Auto-stop. Use model at " << iter-1 << "th iteration." << endl;
                    break;
                } else {
                    best_va_loss = va_loss; 
                }
            }
        }
        cout.width(13);
        cout << fixed << setprecision(1) << timer.get() << endl;
        fm_save_model(model, model_path + "_" + std::to_string(iter));
    }

    return model;
}

fm_int fm_save_model(fm_model &model, string path) {

    ofstream f_out(path);
    if (!f_out.is_open())
        return 1;

    f_out << "n " << model.n << "\n";
    f_out << "k " << model.k << "\n";
    f_out << "bias " << model.w0 << "\n";

    for (fm_int i = 0; i < model.n; i++) {
        f_out << "w " << i << " " << model.weight_map.find(i)->second.w << "\n";
    }
    for (fm_int i = 0; i < model.n; i++) {
        f_out << "v " << i;
        fm_weight_unit &unit = model.weight_map.find(i)->second;
        for (fm_int f = 0; f < model.k; f++) {
            f_out << " " << unit.v.at(f);
        }
        f_out << "\n";
    }
    return 0;
}

fm_model fm_load_model(string path) {

    ifstream f_in(path); // need check

    string dummy;

    fm_model model;

    f_in >> dummy >> model.n
         >> dummy >> model.k
         >> dummy >> model.w0;

    for (fm_int i = 0; i < model.n; i++) {
        fm_weight_unit unit;
        f_in >> dummy;
        f_in >> dummy;
        f_in >> unit.w;
        model.weight_map.insert({i, unit});
    }

    for (fm_int i = 0; i < model.n; i++) {
        fm_weight_unit &unit = model.weight_map.find(i)->second;
        f_in >> dummy;
        f_in >> dummy;
        for (fm_int j = 0; j < model.k; j++) {
            fm_float val = 0;
            f_in >> val;
            unit.v.push_back(val);
        }
    }
    return model;
}

fm_float fm_predict(fm_node *begin, fm_node *end, fm_model &model) {
    fm_float t = wTx(begin, end, model);
    return 1 / (1 + exp(-t));
}

} // namespace fm
