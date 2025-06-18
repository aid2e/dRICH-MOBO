// countScan.cpp
//
// A standalone C++ program (using ROOT) that:
// 1. Takes two command‐line arguments:
//      [input_root_file] [output_text_file]
// 2. Opens the ROOT file, reads the TTree named "tree" and its branch "random_int"
// 3. Counts how many entries have random_int > 5
// 4. Writes that count (as a single integer) into the specified text file and exits

#include <TFile.h>
#include <TTree.h>
#include <TChain.h>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>  // for std::atoi
#include <nlohmann/json.hpp>

int main(int argc, char** argv) {
    // Expect exactly 2 arguments: [input_root_file] [output_text_file]
    if (argc != 6) {
        std::cout << "Usage: " << argv[0]
                  << " [input_root_file] [output_text_file] [output_dir] [Gas] [JSON] \n";
        return 1;
    }

    const char* inputRootPath = argv[1];
    const char* outputTxtPath  = argv[2];
    const char* outputDir = argv[3];
    const char* Gas = argv[4];
    const char* inputJPath = argv[5];

    //Test the JSON File here..
    std::vector<std::string>rfilenames;

    {
        std::string jpath = std::string(inputJPath);
        std::cout<<"Opening the Input JSON File "<<jpath<<std::endl;
        std::ifstream input_jfile(jpath);
        if(!input_jfile.is_open()){
            std::cerr<<" ERROR: Cannot Open JSON File "<<jpath<<std::endl;
        }
        else{
            nlohmann::json js;
            input_jfile>>js;
            try {
                if (js.contains("input_files") && js["input_files"].is_array()) {
                    for (const auto& file : js["input_files"]) {
                        rfilenames.push_back(file.get<std::string>());
                    }
                } else {
                    std::cerr << "Invalid JSON format: 'input_files' key not found or not an array" << std::endl;
                    return 1;
                }
            } catch (const std::exception& ex) {
                std::cerr << "Exception while parsing JSON: " << ex.what() << std::endl;
                return 1;
            }
    
            
        }
    }

    // 1. Open the ROOT file
    /*
    TFile* inFile = TFile::Open(inputRootPath, "READ");
    if (!inFile || inFile->IsZombie()) {
        std::cerr << "ERROR: could not open ROOT file `" << inputRootPath << "`\n";
        return 2;
    }
    */

    TChain chain("tree");

    if(rfilenames.size()!=0){
        for(const auto&file: rfilenames){
            std::cout<<"Adding files to the TChain "<<file<<std::endl;
            chain.Add(file.c_str());
        }
    }
    else{
        chain.Add(inputRootPath);
    }
    
    // 2. Retrieve the TTree named "tree"
    /*
    TTree* tree = nullptr;
    inFile->GetObject("tree", tree);
    if (!tree) {
        std::cerr << "ERROR: TTree \"tree\" not found in `" << inputRootPath << "`\n";
        inFile->Close();
        return 3;
    }
    */
    // 3. Set up a branch address for "random_int"
    Int_t random_int = 0;
    if (chain.SetBranchAddress("random_int", &random_int) < 0) {
        std::cerr << "ERROR: Branch \"random_int\" not found in TTree\n";
        return 4;
    }

    // 4. Loop over all entries and count how many have random_int > 5
    Long64_t nEntries = chain.GetEntries();
    Long64_t count = 0;
    for (Long64_t i = 0; i < nEntries; ++i) {
        chain.GetEntry(i);
        if (random_int > 5) {
            ++count;
        }
    }


    // 6. Open the text file and write the count
   // std::ofstream outFile(outputTxtPath);
    TString outname_wdir = TString(outputDir)+ TString(outputTxtPath);
    FILE *outFile = fopen(outname_wdir.Data(),"w");
  fprintf(outFile, "%lld \n",
	  count);

    std::cout << "Wrote count (" << count << ") to " << outputTxtPath << "\n";
    return 0;
}
