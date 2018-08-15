.PHONY : callgrind
HEADERS = $(wildcard *.h)

default: test.cpp $(HEADERS)
	@g++ -std=c++11 -g -Wall -O2 -I../lib/eigen_3.3.3 -I../lib/boost_1_66_0 -I. test.cpp -o test
