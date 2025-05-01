#include <iostream>
#include <cstring>
#include <string>
#include "cli_config.h"


extern "C" int gpu_crack(const std::string& digest);

int cpu_crack(const std::string& digest, int threads);

void usage(const char *p){
    std::cerr << "Usage: " << p
              << " -d <32-char-digest> [-l 1-7] [-m cpu|gpu] [-t threads]\n";
}

int main(int argc,char **argv){
    std::string digest, mode = "gpu";
    int threads = 0, len = MAX_PW_LEN;

    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"-d") && i+1<argc) digest = argv[++i];
        else if(!strcmp(argv[i],"-l") && i+1<argc) len = std::stoi(argv[++i]);
        else if(!strcmp(argv[i],"-m") && i+1<argc) mode = argv[++i];
        else if(!strcmp(argv[i],"-t") && i+1<argc) threads = std::stoi(argv[++i]);
        else { usage(argv[0]); return 1; }
    }
    if(digest.size()!=32 || len<1 || len>MAX_PW_LEN){
        usage(argv[0]); return 1;
    }
    // normalise
    for(char& c: digest) c = std::tolower(c);

    g_pw_len = len;                          // sets global, will be copied to GPU driver
    if(mode=="cpu") return cpu_crack(digest, threads);
    if(mode=="gpu") return gpu_crack(digest);
    usage(argv[0]); return 1;
}
