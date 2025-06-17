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
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>  // for std::atoi

int main(int argc, char** argv) {
    // Expect exactly 2 arguments: [input_root_file] [output_text_file]
    if (argc != 5) {
        std::cout << "Usage: " << argv[0]
                  << " [input_root_file] [output_text_file] [output_dir] [Gas] \n";
        return 1;
    }

    const char* inputRootPath = argv[1];
    const char* outputTxtPath  = argv[2];
    const char* outputDir = argv[3];
    const char* Gas = argv[4];

    // 1. Open the ROOT file
    TFile* inFile = TFile::Open(inputRootPath, "READ");
    if (!inFile || inFile->IsZombie()) {
        std::cerr << "ERROR: could not open ROOT file `" << inputRootPath << "`\n";
        return 2;
    }

    // 2. Retrieve the TTree named "tree"
    TTree* tree = nullptr;
    inFile->GetObject("tree", tree);
    if (!tree) {
        std::cerr << "ERROR: TTree \"tree\" not found in `" << inputRootPath << "`\n";
        inFile->Close();
        return 3;
    }

    // 3. Set up a branch address for "random_int"
    Int_t random_int = 0;
    if (tree->SetBranchAddress("random_int", &random_int) < 0) {
        std::cerr << "ERROR: Branch \"random_int\" not found in TTree\n";
        inFile->Close();
        return 4;
    }

    // 4. Loop over all entries and count how many have random_int > 5
    Long64_t nEntries = tree->GetEntries();
    Long64_t count = 0;
    for (Long64_t i = 0; i < nEntries; ++i) {
        tree->GetEntry(i);
        if (random_int > 5) {
            ++count;
        }
    }

    // 5. Close the ROOT file
    inFile->Close();

    // 6. Open the text file and write the count
   // std::ofstream outFile(outputTxtPath);
    TString outname_wdir = TString(outputDir)+ TString(outputTxtPath);
    FILE *outFile = fopen(outname_wdir.Data(),"w");
  fprintf(outFile, "%lld \n",
	  count);

    std::cout << "Wrote count (" << count << ") to " << outputTxtPath << "\n";
    return 0;
}
