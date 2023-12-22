
# Compiler
CXX := g++
# Compiler flags
CXXFLAGS := -Wall -Wno-deprecated -I$(shell root-config --incdir) $(shell root-config --glibs) -L/usr/local/lib -lpodio -lpodioRootIO -ledm4hep -ledm4eic

dRICHAna: dRICHAna.cpp
	$(CXX) -o dRICHAna dRICHAna.cpp $(CXXFLAGS)

# Rule to clean the project
clean:
	rm -f $(OBJS) $(TARGET)
