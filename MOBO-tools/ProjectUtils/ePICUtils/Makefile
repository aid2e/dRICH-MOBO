# Compiler
CXX := g++
# Compiler flags
CXXFLAGS := -Wall -Wno-deprecated -I$(shell root-config --incdir) $(shell root-config --glibs) -L/opt/local/lib -lpodio -lpodioRootIO -ledm4hep -ledm4eic -I/opt/local/include

# Executables
TARGETS := dRICHAna dRICHAna_all dRICHAna_fake

.PHONY: all clean

# Default rule: builds all targets
all: $(TARGETS)

%: %.cpp
	$(CXX) -o $@ $< $(CXXFLAGS)

#dRICHAna: dRICHAna.cpp
#	$(CXX) -o dRICHAna dRICHAna.cpp $(CXXFLAGS)

#dRICHAna_all: dRICHAna_all.cpp
#	$(CXX) -o dRICHAna_all dRICHAna_all.cpp $(CXXFLAGS)

# Rule to clean the project
clean:
	rm -f $(TARGETS)
