.PHONY : callgrind
HEADERS = $(wildcard *.h)

default: test.cpp $(HEADERS)
	@g++ -std=c++11 -g -Wall -O0 -I../lib/eigen_3.3.3 -I../lib/boost_1_66_0 -I../lib/LBFGSpp/include -I. test.cpp -o test
