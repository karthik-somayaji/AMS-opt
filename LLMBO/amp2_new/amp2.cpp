#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string.h>
#include <sstream>
#include <vector>
#ifndef SERIAL
#include <omp.h>
#endif

using namespace std;
const int f_num = 14;
const string amp = "amp2_";

int sim_gen(const string& id, double* s){
    string cmd = "echo '* Sim\n.inc ./param" + id + ".inc' > amp2_" + id + ".sp && cat amp2.sp >> amp2_" + id + ".sp";
    system(cmd.c_str());

    string file = "param" + id + ".inc";
    ofstream fout(file.c_str(), ios_base::out);
    for (int i = 0; i < 6; ++i) fout << ".param l" << i+1 << " = " << s[i] << endl;
    for (int i = 0; i < 6; ++i) fout << ".param w" << i+1 << " = " << s[i+6] << endl;
    fout << ".param c = " << s[12] << endl;
    fout << ".param r = " << s[13] << endl;
    fout.close();
    return 0;
}

int read_vol(const string& id, vector<double>& vol) {
    string filename = amp + id + ".ic1";
    ifstream fin(filename.c_str(), ios_base::in);
    string line;
    vol.resize(6);
    while(getline(fin, line))
        if(line == ".nodeset") break;
    int cnt = 0;
    while(getline(fin, line)) {
        istringstream iss(line);
        string plus, equal;
        string node;
        double v;
        iss >> plus >> node >> equal >> v;
        if(node == "node1") {vol[0] = v; ++cnt;}
        else if(node == "node2") {vol[1] = v; ++cnt;}
        else if(node == "out") {vol[2] = v; ++cnt;}
        else if(node == "out1") {vol[3] = v; ++cnt;}
        else if(node == "outm") {vol[4] = v; ++cnt;}
        else if(node == "vb") {vol[5] = v; ++cnt;}
        if(cnt == 6) break;
    }
    fin.close();
    return 1;
}

int read_op(const string& id, double& pwr, vector< vector<double> >& op) {
    string filename = amp + id + ".dp1";
    ifstream fin(filename.c_str(), ios_base::in);
    op.resize(8);
    string line;
    while(getline(fin, line))
        if(line.substr(0, 14) == " element v_sup") break;
    getline(fin, line);
    getline(fin, line);
    getline(fin, line);
    istringstream isstemp(line);
    string power;
    isstemp >> power >> pwr;
    while(getline(fin, line))
        if(line == " **** mosfets") break;
    for(int i = 0; i < 8; ++i) {
        getline(fin, line);
        getline(fin, line);
        getline(fin, line);
        getline(fin, line);
        getline(fin, line);
        // Region
        getline(fin, line);
        istringstream issregion(line);
        string region;
        string region_val;
        issregion >> region >> region_val;
        if(region_val == "Subth") op[i].push_back(0.0);
        else if(region_val == "Saturation") op[i].push_back(1.0);
        else if(region_val == "Linear") op[i].push_back(-1.0);
        else if(region_val == "Cutoff") op[i].push_back(-2.0);
        // Val
        for(int j = 0; j < 20; ++j) {
            getline(fin, line);
            istringstream iss(line);
            string name;
            double v;
            iss >> name >> v;
            op[i].push_back(v);
        }
        getline(fin, line);
    }
    
    
    fin.close();
    return 1;
}

int read_meas(const string& id, double& gbw, double& gain, double &cmrr, double& pm){
    double acm;
    string filename1 = amp + id + ".ma0";
    ifstream fin1(filename1.c_str(), ios_base::in);
    char line1[256];
    fin1.getline(line1,256);
    fin1.getline(line1,256);
    fin1.getline(line1,256);
    fin1>>acm;
    fin1.close();
    string filename2 = amp + id + ".ma1";
    ifstream fin2(filename2.c_str(), ios_base::in);
    char line2[256];
    fin2.getline(line2,256);
    fin2.getline(line2,256);
    fin2.getline(line2,256);
    fin2.getline(line2,256);
    fin2>>gain>>gbw>>pm;
    fin2.close();
    cmrr=gain-acm;
    return 1;
}

int read_res(const string& id, double& gbw, double& gain, double& cmrr, double& pm, double& pwr, vector<double>& vol, vector< vector<double> >& op) {
    int meas_res = read_meas(id, gbw, gain, cmrr, pm);
    int vol_res = read_vol(id, vol);
    int op_res = read_op(id, pwr, op);
    return meas_res*vol_res*op_res;
}

int run_sim(const string& id){
    string cmd = "hspice ";
    cmd += amp + id + ".sp > out";
    cmd += id + ".txt";
    int res = system(cmd.c_str());
    return res;
}

void run_all(int id, double* s, double& gbw, double& gain, double& cmrr, double& pm, double& pwr, vector<double>& vol, vector< vector<double> >& op)
{
    stringstream ss; ss << id;
    string str_id = ss.str();
    int state = sim_gen(str_id,s);
    if (state==0){
        state = run_sim(str_id);
        if (state==0) state = read_res(str_id, gbw, gain, cmrr, pm, pwr, vol, op);
    }
}

int main(int argc, char* argv[]){
    if(argc != 2){
        cout << "input argument error" << endl;
        exit(-1);
    }
    string x_path = argv[1];
    const char* xfile = "x_added.dat";
    const char* yfile = "y_added.dat";
#ifndef SERIAL
    const int THREAD_NUM = 5;
#else
    const int THREAD_NUM = 1;
#endif
    ofstream xout(xfile, ios_base::out);
    ofstream yout(yfile, ios_base::out);

    long counter = 0;
    ifstream xin(x_path.c_str(), ios_base::in);

    double *s[THREAD_NUM];
    for(int i = 0; i < THREAD_NUM; ++i)
        s[i] = new double[f_num];
    double gbw[THREAD_NUM], gain[THREAD_NUM], cmrr[THREAD_NUM], pm[THREAD_NUM], pwr[THREAD_NUM];
    vector<double> vol[THREAD_NUM];
    vector< vector<double> > op[THREAD_NUM];
    int count = 0;

    int id;

    string temp;
    stringstream line;
    while(getline(xin, temp)){
        line << temp;
        for (int i=0;i<f_num;++i) {line>>s[count][i];};
        line.clear();
        line.str("");
        ++counter;
        if(++count == THREAD_NUM)
        {
#ifndef SERIAL
#pragma omp parallel for default(none) shared(gbw, gain, cmrr, pm, pwr, vol, op, s, count) private(id)
#endif
            for(id = 0; id < count; ++id)
            {
                run_all(id, s[id], gbw[id], gain[id], cmrr[id], pm[id], pwr[id], vol[id], op[id]);
            }
            for(id = 0; id < count; ++id)
            {
                yout<<setiosflags(ios::scientific)<<setprecision(10);
                cout<<setiosflags(ios::scientific)<<setprecision(10);
                yout<<gbw[id] <<' '<<gain[id]<<' '<<cmrr[id]<<' '<<pm[id]<<' '<<pwr[id];
                cout<<gbw[id] <<' '<<gain[id]<<' '<<cmrr[id]<<' '<<pm[id]<<' '<<pwr[id];
                for(int i = 0; i < vol[id].size(); ++i) {
                    yout<<' '<<vol[id][i];
                    cout<<' '<<vol[id][i];
                }
                for(int i = 0; i < op[id].size(); ++i) {
                    for(int j = 0; j < op[id][i].size(); ++j) {
                        yout<<' '<<op[id][i][j];
                        cout<<' '<<op[id][i][j];
                    }
                }
                yout << endl;
                cout << endl;
                xout<<setiosflags(ios::scientific)<<setprecision(10);
                for (int j=0;j<f_num;++j) xout<<s[id][j]<<" ";
                xout<<endl;
            }
            count = 0;
        }
    }
#ifndef SERIAL
#pragma omp parallel for default(none) shared(gbw, gain, cmrr, pm, pwr, vol, op, s, count) private(id)
#endif
    for(id = 0; id < count; ++id)
    {
        run_all(id, s[id], gbw[id], gain[id], cmrr[id], pm[id], pwr[id], vol[id], op[id]);
    }
    for(id = 0; id < count; ++id)
    {
        yout<<setiosflags(ios::scientific)<<setprecision(10);
        cout<<setiosflags(ios::scientific)<<setprecision(10);
        yout<<gbw[id] <<' '<<gain[id]<<' '<<cmrr[id]<<' '<<pm[id]<<' '<<pwr[id];
        cout<<gbw[id] <<' '<<gain[id]<<' '<<cmrr[id]<<' '<<pm[id]<<' '<<pwr[id];
        for(int i = 0; i < vol[id].size(); ++i) {
            yout<<' '<<vol[id][i];
            cout<<' '<<vol[id][i];
        }
        for(int i = 0; i < op[id].size(); ++i) {
            for(int j = 0; j < op[id][i].size(); ++j) {
                yout<<' '<<op[id][i][j];
                cout<<' '<<op[id][i][j];
            }
        }
        yout << endl;
        cout << endl;
        xout<<setiosflags(ios::scientific)<<setprecision(10);
        for (int j=0;j<f_num;++j) xout<<s[id][j]<<" ";
        xout<<endl;
    }
    xout.close();
    yout.close();
    xin.close();
    for(int i = 0; i < 5; ++i)
        delete[] s[i];
    return 0;
}
