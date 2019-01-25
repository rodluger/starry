.PHONY: default clean
HEADERS = $(wildcard *.h) $(wildcard ../include/starry2/*.h) $(wildcard ../include/starry2/*/*.h) $(wildcard ../include/starry2/*/*/*.h)

default: default.cpp $(HEADERS)
	g++ -std=c++11 -g -Wall -O0 -I../starry2 -I../../lib/eigen_3.3.5 -I. default.cpp -o default

spectral: spectral.cpp $(HEADERS)
	g++ -std=c++11 -g -Wall -O0 -I../starry2 -I../../lib/eigen_3.3.5 -I. spectral.cpp -o spectral

temporal: temporal.cpp $(HEADERS)
	g++ -std=c++11 -g -Wall -O0 -I../starry2 -I../../lib/eigen_3.3.5 -I. temporal.cpp -o temporal

clean:
	@rm -rf default
	@rm -rf spectral
	@rm -rf temporal
	@rm -rf *.dSYM