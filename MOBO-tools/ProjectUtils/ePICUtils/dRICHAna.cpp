#include "edm4hep/MCParticleCollection.h"
#include "edm4eic/CherenkovParticleIDCollection.h"
#include "podio/ROOTFrameReader.h"
#include "podio/Frame.h"
#include "TH1F.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TString.h"
using namespace std;


void extractSPEres(const char* filename, const char* outname, const char* outdir, int radiator){

  double thlow, thhigh;
  int nbins;
  if(radiator==0){
    thlow = 150;
    thhigh = 220;
    nbins = 200;
  }
  else{
    thlow = 20;
    thhigh = 60;
    nbins = 100;
  }
  TH1F* hSingleTheta = new TH1F("hSingleTheta", "", nbins, thlow, thhigh);
  TH1F* hnPhotons = new TH1F("hnPhotons", "", 60, -0.5, 60.5);  
  
  podio::ROOTFrameReader reader;
  reader.openFile(filename);
  
  int nev = reader.getEntries("events");
    
  // event loop
  for(int i = 0; i < nev; i++){
    const auto event = podio::Frame(reader.readNextEntry("events"));

    std::string pidCollection;
    double n;
    if(radiator==0){
      pidCollection = "DRICHAerogelIrtCherenkovParticleID";
      n = 1.01826;
    }
    else{
      pidCollection = "DRICHGasIrtCherenkovParticleID";
      n = 1.000746;
    }
    
    auto& dRichCherenkov = event.get<edm4eic::CherenkovParticleIDCollection>(pidCollection);
    auto& MCParticles = event.get<edm4hep::MCParticleCollection>("MCParticles");    

    double px, py, pz, p, mass;
    double betaTrue;    

    // need to check MC particles, see if we missed anything (account for efficiency)
    if(MCParticles.isValid()){
      px = MCParticles[0].getMomentum().x;
      py = MCParticles[0].getMomentum().y;
      pz = MCParticles[0].getMomentum().z;
      p = sqrt(px*px+py*py+pz*pz);
      mass = MCParticles[0].getMass();
      betaTrue = p/(sqrt(p*p+mass*mass)); 
    }
    else{
      cout << "Error: no thrown particles" << endl;
      continue;
    }
    
    if (dRichCherenkov.isValid()) {
      double chExpected = acos(1/(n*betaTrue))*1000;
      
      for(unsigned int j = 0; j < dRichCherenkov.size(); j++){
	auto thetaPhi = dRichCherenkov[j].getThetaPhiPhotons();
	int nPhotons = (int)thetaPhi.size();
	hnPhotons->Fill(nPhotons);

	for(int k = 0; k < thetaPhi.size(); k++){
	  hSingleTheta->Fill(thetaPhi[k][0]*1000);
	}
      }      
    }
    else{
      hnPhotons->Fill(0);
    }
    
  }
  double mean = hSingleTheta->GetMean();
  double rms = hSingleTheta->GetRMS();
  
  TF1 *f1 = new TF1("gaussianFit", "gaus", mean-2*rms, mean+2*rms);  
  hSingleTheta->Fit("gaussianFit","R");
  TCanvas *c = new TCanvas();
  hSingleTheta->Draw();
  c->SaveAs("spetest.png");

  TString outname_wdir = TString(outdir) + TString(outname);
  FILE *outfile = fopen(outname_wdir.Data(),"w");

  
  fprintf(outfile, "%lf %lf %lf %lf \n",
	  hnPhotons->GetMean(), f1->GetParameter(1),
	  f1->GetParameter(2), f1->GetChisquare());
  
  return;
}


int main(int argc, char* argv[]){
  if(argc < 1){
    cout << "usage: dRICHAana [filename] [outputname (txt)] [output dir] [radiator: 0 - aerogel, 1 - gas] \n";
    return 1;
  }
  int rad;
  stringstream s(argv[4]);
  s >> rad;
  
  extractSPEres(argv[1], argv[2], argv[3], rad);  
  return 0;  
}
