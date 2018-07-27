.PHONY : callgrind
HEADERS = $(wildcard *.h)

default: test.cpp $(HEADERS)
	@g++ -std=c++11 -Wall -I../lib/eigen_3.3.3 -I../lib/boost_1_66_0 -I../lib/LBFGSpp/include -I. test.cpp -o test

callgrind: test.cpp $(HEADERS)
	@g++ -std=c++11 -Wall -g -I../lib/eigen_3.3.3 -I../lib/boost_1_66_0 -I../lib/LBFGSpp/include -I. test.cpp -o test
	@valgrind --tool=callgrind --dump-instr=yes --collect-jumps=yes ./test
	@qcachegrind

clean:
	@rm -rf callgrind.out.*
	@rm -rf *.dSYM
